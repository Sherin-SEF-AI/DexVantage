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
DATASET_STUDIES = ('audit', 'plan_vs_recording', 'silhouette_calib')     # studies that read --dataset
# the flag names of the specification (§11) where a study's own flag differs: both work
BATCH_ALIASES = {'teleop': ('teleop-overlap',), 'time_sync': ('sync',)}


def add_arguments(ap):
    g = ap.add_argument_group('batch mode (no window)')
    g.add_argument('--analyse', metavar='OUT.json', help='write the project summary study as JSON ("-" for stdout)')
    g.add_argument('--validate', metavar='OUT_DIR', help='run the validation suite and write its report')
    g.add_argument('--models', help='with --validate: comma-separated model ids (default: all)')
    g.add_argument('--quick', action='store_true', help='with --validate: fewer Monte Carlo trials (a smoke run)')
    g.add_argument('--reproduce', metavar='RESULT.json', help='re-run the study that wrote RESULT.json and compare')
    g.add_argument('--tolerance', action='append', metavar='[PATTERN=]VALUE[%]',
                   help='with --reproduce: allowed differences, e.g. 1e-9 or voxels*=0')
    g.add_argument('--synth-episodes', metavar='OUT_DIR', help='generate synthetic LeRobot episodes with known faults '
                   'and their ground truth into a new or empty folder (core/sim/episodes.py)')
    g.add_argument('--synth-settings', metavar='JSON', help='with --synth-episodes: settings as JSON text or a .json '
                   'file, e.g. {"episodes": 2, "version": "v2.1"}')
    g.add_argument('--dataset', metavar='PATH', help='with --audit, --plan-vs-recording or --silhouette-calib: the '
                   'recorded dataset (a LeRobot folder, an RLDS folder or a ROS 2 bag)')
    g.add_argument('--audit-html', metavar='OUT.html', help='with --audit: also write the audit report as a standalone '
                   'HTML page (core/audit/report.py)')
    g.add_argument('--fiftyone', metavar='OUT_DIR', help='with --audit: also export the per-episode audit scores as a '
                   'FiftyOne dataset (sample tags and fields; a new or empty folder)')
    from .core import jobs
    jobs.load_studies()
    s = ap.add_argument_group('studies (each writes its result as JSON with provenance; "-" for stdout)')
    for info in jobs.STUDIES.values():
        if info.batch:
            s.add_argument(f'--{info.batch}', *[f'--{a}' for a in BATCH_ALIASES.get(info.id, ())], metavar='OUT.json',
                           dest=f'study__{info.id}', help=info.title)


def _study_flags(args) -> list:
    return [(k[len('study__'):], v) for k, v in vars(args).items() if k.startswith('study__') and v]


def wanted(args) -> bool:
    return bool(args.analyse or args.validate or args.reproduce or _study_flags(args)
                or getattr(args, 'synth_episodes', None) or getattr(args, 'audit_html', None)
                or getattr(args, 'fiftyone', None) or getattr(args, 'dataset', None))


def _load(args):
    from .core.project import Project, ProjectFormatError
    if getattr(args, '_project', None) is not None:     # --sample (app.py)
        return args._project
    if not args.path:
        return None
    try:
        return Project.load(args.path)
    except ProjectFormatError as e:
        raise SystemExit(_err(str(e)))


def _err(text) -> int:
    print(f'dexvantage: {text}', file=sys.stderr)
    return EXIT_ERROR


def _unexpected(title: str, e: Exception) -> int:
    """An error no study explains: one sentence on stderr, the traceback in a file (or on stderr with
    DEXVANTAGE_DEBUG=1)."""
    import tempfile
    import traceback
    first = (str(e).splitlines() or [''])[0][:300]
    if os.environ.get('DEXVANTAGE_DEBUG'):
        traceback.print_exc()
        where = 'above'
    else:
        fd, where = tempfile.mkstemp(prefix='dexvantage-error-', suffix='.txt')
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            traceback.print_exc(file=fh)
    return _err(f'{title} stopped with an unexpected error ({type(e).__name__}: {first}). This is a bug or an input '
                f'DexVantage does not expect; the details are in {where}.')


def _write(doc, dest):
    text = json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False) + '\n'
    if dest in (None, '', '-'):
        sys.stdout.write(text)
        return
    tmp = dest + '.part'
    try:
        with open(tmp, 'w', encoding='utf-8') as fh:
            fh.write(text)
        os.replace(tmp, dest)
    except OSError as e:                  # a folder that does not exist, no permission, a full disk: say so, exit 2
        raise SystemExit(_err(f'could not write {dest}: {e.strerror or e}. Choose a folder that exists and that you '
                              'can write to.'))


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
        if (getattr(args, 'audit_html', None) or getattr(args, 'fiftyone', None)) and \
                'audit' not in [sid for sid, _ in _study_flags(args)]:
            return _err('--audit-html and --fiftyone go with --audit OUT.json (and --dataset PATH)')
        if getattr(args, 'dataset', None) and not any(sid in DATASET_STUDIES for sid, _ in _study_flags(args)):
            return _err('--dataset goes with --audit, --plan-vs-recording or --silhouette-calib OUT.json')
        if args.analyse:
            if project is None:
                return _err('--analyse needs a project file')
            doc = _summary(project, args.analyse)
            _write(doc, args.analyse)
            if args.analyse != '-':
                vols = doc['volumes']
                print(f'{project.name}: {len(vols)} task volume{"" if len(vols) == 1 else "s"}, '
                      f'{sum(v["voxels_dense"] for v in vols):,} voxels at {doc["voxel_mm"]:g} mm → {args.analyse}')
        for sid, dest in _study_flags(args):
            from .core import jobs
            snap = project.snapshot() if project is not None else None
            if sid in DATASET_STUDIES and getattr(args, 'dataset', None):
                from .core.project import blank_project
                from .core.studies.audit import use_dataset
                snap = snap if snap is not None else (blank_project('No project') if sid == 'audit' else None)
                if snap is not None:
                    use_dataset(snap, args.dataset, sid)
            if snap is None:
                return _err(f'--{jobs.STUDIES[sid].batch} needs a project file')
            try:
                doc = jobs.run_study(sid, snap)
            except jobs.StudyError as e:
                return _err(str(e))
            except Exception as e:                      # noqa: BLE001 — a sentence here, the traceback in a file
                return _unexpected(jobs.STUDIES[sid].title, e)
            doc = {k: v for k, v in doc.items() if k != 'cached'}
            option = jobs.STUDIES[sid].batch.replace('-', '_')
            doc = PV.stamp(doc, project if project is not None else snap,
                           output=PV.batch_output(doc, dest, option=option))
            _write(doc, dest)
            if dest != '-':
                print(f'{jobs.STUDIES[sid].title} → {dest}')
            if sid == 'audit':
                code = max(code, _audit_exports(args, doc, project if project is not None else snap, snap))
        if getattr(args, 'synth_episodes', None):
            from .core import jobs
            from .core.sim.episodes import batch_generate
            if project is None:
                return _err('--synth-episodes needs a project file (or --sample)')
            try:
                print(batch_generate(project, args.synth_episodes, args.synth_settings))
            except jobs.StudyError as e:
                return _err(str(e))
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


def _audit_exports(args, doc, project, snap) -> int:
    """--audit-html and --fiftyone after --audit: the HTML report and the FiftyOne export of the audit just run."""
    from .core import jobs
    from .core.audit import report as RP
    from .core.studies.audit import dataset_path
    if getattr(args, 'audit_html', None):
        try:
            RP.write_html(doc, project, args.audit_html)
        except OSError as e:
            return _err(f'could not write {args.audit_html}: {e.strerror or e}. Choose a folder that exists and that '
                        'you can write to.')
        print(f'Dataset audit report → {args.audit_html}')
    if getattr(args, 'fiftyone', None):
        try:
            out = RP.export_fiftyone(doc, dataset_path(snap), args.fiftyone, snap)
        except jobs.StudyError as e:
            return _err(str(e))
        except OSError as e:
            return _err(f'could not write {args.fiftyone}: {e.strerror or e}.')
        print(f'FiftyOne export ({out.get("samples", "?")} samples) → {args.fiftyone}')
    return EXIT_OK


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
    from .core import jobs
    jobs.load_studies()
    if study not in jobs.STUDIES:
        rep = PV.report(os.path.basename(args.reproduce), recorded, None, tol,
                        error=f'this build cannot re-run the study "{study}" yet')
    elif project is None:
        return _err('--reproduce needs the project file the result was made from')
    else:
        doc = _summary(project, '-') if study == 'summary' else PV.stamp(
            {k: v for k, v in jobs.run_study(study, project.snapshot()).items() if k != 'cached'}, project)
        rep = PV.report(os.path.basename(args.reproduce), recorded, doc, tol, now_provenance=doc[PV.KEY],
                        project_now=PV.project_digest(project))
    print(rep['message'] if 'message' in rep else rep.get('verdict_text', ''))
    for r in rep.get('reasons') or []:
        print(f'  · {r}')
    return EXIT_OK if rep['verdict'] in ('reproduces', 'reproduces_within_tolerance') else EXIT_FAIL
