"""Batch mode: DexVantage without a window.

    dexvantage PROJECT.dexv --analyse summary.json      the project summary (task volumes, voxel counts)
    dexvantage [PROJECT.dexv] --validate OUT_DIR        the validation suite and its report (JSON, CSV, HTML, figures)
    dexvantage PROJECT.dexv --reproduce RESULT.json     re-run what produced RESULT.json and compare every number
    dexvantage PROJECT.dexv --STUDY-FLAG OUT.json       any study (its flag is listed by --help), e.g. --coverage,
                                                        --tolerance-mc, --optimise [--optimise-settings JSON]
    dexvantage PROJECT.dexv --experiment DESIGN.json    a designed experiment (core/experiments.py)
    dexvantage PROJECT.dexv --campaign SPEC.json        a data-collection campaign plan (batch_campaign.py)
    dexvantage [PROJECT.dexv] --stats SPEC.json         statistics and power analysis (batch_stats.py)
    dexvantage PROJECT.dexv --splat fit|replay|augment --splat-out OUT_DIR [--dataset PATH]   splat views
                                                        (batch_splat.py; needs the splat extra)
    dexvantage PROJECT.dexv --check REQUIREMENTS.yaml   the readiness scorecard CI gate; --run-plan PLAN.json
                                                        (batch_gate.py)
    dexvantage PROJECT.dexv --export OUT_DIR --formats A,B   the exporters (core/exporters), with a manifest
    dexvantage PROJECT.dexv --rerun OUT.rrd             the scene for Rerun (core/bridges/rerun.py)
    dexvantage A.dexv --diff B.dexv | --merge | --unpack | --pack | --textconv | --cite | --figure
                                                        research tools (batch_research.py)

'-' as an output writes to standard output; a study given a folder (one that exists, or a path ending in '/', as in
the specification's `--mujoco-validate out/`) writes <folder>/<flag>.json. Exit codes: 0 done (validation: every
check passed; reproduce: the numbers reproduce), 1 a failed check or a result that does not reproduce, 2 an error
that stopped the run (a file that cannot be read, a bad option). Each study registered with a batch flag
(core/jobs.study) gets its flag here; the spec's other names are aliases (BATCH_ALIASES: --grasp-study, --cable-flex,
--mujoco-validate, --teleop-overlap, --sync). `--handeye-solve OUT.json --captures DIR [--joints FILE]` captures a
folder of board images at run time (the specification's `--handeye-solve captures/`).

Ported from Vantage batch.py (the structure and the provenance handling).
"""
from __future__ import annotations

import json
import os
import sys

EXIT_OK, EXIT_FAIL, EXIT_ERROR = 0, 1, 2
DATASET_STUDIES = ('audit', 'plan_vs_recording', 'silhouette_calib', 'sim2real',
                   'splat_views')         # studies that read --dataset
# the flag names of the specification (§11) where a study's own flag differs: both work
OPTIMISE_STUDIES = ('optimise', 'design_agent')     # studies that take --optimise-settings
BATCH_ALIASES = {'teleop': ('teleop-overlap',), 'time_sync': ('sync',), 'grasp_visibility': ('grasp-study',),
                 'cables': ('cable-flex',), 'groundtruth': ('mujoco-validate',)}
JOINTS_FILES = ('joint_states.csv', 'joint_states.json', 'joints.csv', 'joints.json')   # --captures: found beside


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
    g.add_argument('--dataset', metavar='PATH', help='with --audit, --plan-vs-recording, --silhouette-calib or '
                   '--sim2real (or --export with the format fiftyone): the recorded dataset (a LeRobot folder, an '
                   'RLDS folder or a ROS 2 bag)')
    g.add_argument('--captures', metavar='DIR', help='with --handeye-solve: a folder of board images to capture at '
                   'run time; the joint states are its joint_states.csv or .json unless --joints names the file '
                   '(core/calib/capture.py)')
    g.add_argument('--joints', metavar='FILE', help='with --captures: the joint_states file (CSV or JSON)')
    g.add_argument('--audit-html', metavar='OUT.html', help='with --audit: also write the audit report as a standalone '
                   'HTML page (core/audit/report.py)')
    g.add_argument('--fiftyone', metavar='OUT_DIR', help='with --audit: also export the per-episode audit scores as a '
                   'FiftyOne dataset (sample tags and fields; a new or empty folder)')
    g.add_argument('--optimise-settings', metavar='JSON', help='with --optimise or --design-agent: settings as JSON '
                   'text or a .json file, merged over the project\'s, e.g. {"budget": 200, "seed": 3}')
    g.add_argument('--rerun', metavar='OUT.rrd', help='save the scene for Rerun: the robot over its trajectory, camera '
                   'frusta, min-Z blind zones, task volumes and, with --audit or --rerun-audit, the audit flags per '
                   'episode (core/bridges/rerun.py; needs the optional extra rerun)')
    g.add_argument('--rerun-audit', metavar='AUDIT.json', help='with --rerun: an audit result written before by '
                   '--audit')
    g.add_argument('--experiment', metavar='DESIGN.json', help='run a designed experiment (schema '
                   'dexvantage.experiment/1, docs/models/experiments.md) on the project; writes the result JSON with '
                   'provenance and the runs as CSV (core/experiments.py)')
    g.add_argument('--experiment-out', metavar='OUT.json', help='with --experiment: the result file (default '
                   'DESIGN.results.json beside the design; the CSV goes beside it with .csv)')
    g.add_argument('--export', metavar='OUT_DIR', help='export the project with --formats (core/exporters); writes '
                   'each format\'s files and export_manifest.json (SHA-256 of every file, provenance)')
    g.add_argument('--formats', metavar='A,B,...', help='with --export: comma-separated export formats, e.g. '
                   'mjcf,urdf,opencv_yaml,masks,plucker,brackets')
    g.add_argument('--export-options', metavar='JSON', help='with --export: options per format as JSON text or a .json '
                   'file, e.g. {"mjcf": {"mode": "snippets"}, "masks": {"atlas": false}}')
    from . import batch_gate
    batch_gate.add_arguments(g)                         # --check, --reproduce-out, --run-plan (batch_gate.py)
    from . import batch_research
    batch_research.add_arguments(g)                     # --diff, --merge, --unpack, --pack, --textconv, --cite …
    from . import batch_campaign
    batch_campaign.add_arguments(g)                     # --campaign SPEC.json [--campaign-out OUT.json]
    from . import batch_track
    batch_track.add_arguments(g)                        # --track, --runs-export (core/tracking)
    from . import batch_cells
    batch_cells.add_arguments(g)                        # --cells A.dexv B.dexv … (batch_cells.py)
    from . import batch_cluster
    batch_cluster.add_arguments(g)                      # --jobs, --slurm, --run-shard, --gather (core/cluster)
    from . import batch_stats
    batch_stats.add_arguments(g)                        # --stats SPEC.json [--stats-out OUT.json] (batch_stats.py)
    from . import batch_splat
    batch_splat.add_arguments(g)                        # --splat fit|replay|augment --splat-out OUT_DIR
    from .core import jobs
    jobs.load_studies()
    s = ap.add_argument_group('studies (each writes its result as JSON with provenance; "-" for stdout; a folder '
                              'gets <folder>/<flag>.json)')
    for info in jobs.STUDIES.values():
        if info.batch:
            s.add_argument(f'--{info.batch}', *[f'--{a}' for a in BATCH_ALIASES.get(info.id, ())], metavar='OUT.json',
                           dest=f'study__{info.id}', help=info.title)


def _study_flags(args) -> list:
    return [(k[len('study__'):], v) for k, v in vars(args).items() if k.startswith('study__') and v]


def wanted(args) -> bool:
    return bool(args.analyse or args.validate or args.reproduce or _study_flags(args)
                or getattr(args, 'synth_episodes', None) or getattr(args, 'audit_html', None)
                or getattr(args, 'fiftyone', None) or getattr(args, 'dataset', None)
                or getattr(args, 'experiment', None) or getattr(args, 'experiment_out', None)
                or getattr(args, 'export', None) or getattr(args, 'formats', None)
                or getattr(args, 'export_options', None)
                or getattr(args, 'optimise_settings', None)
                or getattr(args, 'captures', None) or getattr(args, 'joints', None)
                or getattr(args, 'rerun', None) or getattr(args, 'rerun_audit', None)
                or __import__('dexvantage.batch_gate', fromlist=['wanted']).wanted(args)
                or __import__('dexvantage.batch_research', fromlist=['wanted']).wanted(args)
                or __import__('dexvantage.batch_campaign', fromlist=['wanted']).wanted(args)
                or __import__('dexvantage.batch_track', fromlist=['wanted']).wanted(args)
                or __import__('dexvantage.batch_cells', fromlist=['wanted']).wanted(args)
                or __import__('dexvantage.batch_cluster', fromlist=['wanted']).wanted(args)
                or __import__('dexvantage.batch_stats', fromlist=['wanted']).wanted(args)
                or __import__('dexvantage.batch_splat', fromlist=['wanted']).wanted(args))


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


def study_dest(dest, flag: str):
    """A study's output path: a folder (one that exists, or a path ending in a separator, as in the specification's
    `--mujoco-validate out/`) gets <folder>/<flag>.json; '-' and file names are kept."""
    if dest in (None, '', '-'):
        return dest
    if os.path.isdir(dest) or dest.endswith(('/', os.sep)):
        try:
            os.makedirs(dest, exist_ok=True)
        except OSError as e:
            raise SystemExit(_err(f'could not create the folder {dest}: {e.strerror or e}.'))
        return os.path.join(dest, f'{flag}.json')
    return dest


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
    if getattr(args, 'textconv', None):                 # git's textconv driver: the text alone (batch_research.py)
        from . import batch_research
        return batch_research.textconv(args)
    try:
        project = _load(args)
    except SystemExit as e:
        return int(e.code)
    if args.reproduce:
        return _reproduce(args, project)
    from . import batch_cluster
    if batch_cluster.wanted(args):                      # --jobs, --slurm, --run-shard, --gather (batch_cluster.py)
        return batch_cluster.run(args, project)
    ctx = PV.begin_batch(args, parser) if parser is not None else None
    try:
        if project is not None:
            PV.note_project(project)
        code = EXIT_OK
        if (getattr(args, 'audit_html', None) or getattr(args, 'fiftyone', None)) and \
                'audit' not in [sid for sid, _ in _study_flags(args)]:
            return _err('--audit-html and --fiftyone go with --audit OUT.json (and --dataset PATH)')
        if getattr(args, 'dataset', None) and not any(sid in DATASET_STUDIES for sid, _ in _study_flags(args)) \
                and not getattr(args, 'run_plan', None) and 'fiftyone' not in _formats(args) \
                and not getattr(args, 'splat', None):
            return _err('--dataset goes with --audit, --plan-vs-recording, --silhouette-calib, --sim2real or '
                        '--splat-views OUT.json (or --splat ACTION, or --export with the format fiftyone)')
        if (getattr(args, 'captures', None) or getattr(args, 'joints', None)) and \
                'handeye_solve' not in [sid for sid, _ in _study_flags(args)]:
            return _err('--captures DIR (and --joints FILE) go with --handeye-solve OUT.json')
        if getattr(args, 'joints', None) and not getattr(args, 'captures', None):
            return _err('--joints FILE goes with --captures DIR')
        if getattr(args, 'optimise_settings', None) and \
                not any(sid in OPTIMISE_STUDIES for sid, _ in _study_flags(args)):
            return _err('--optimise-settings goes with --optimise or --design-agent OUT.json')
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
            if sid == 'handeye_solve' and getattr(args, 'captures', None):
                try:
                    snap.analysis.studies[sid] = {**(snap.analysis.studies.get(sid) or {}), **_captures(args)}
                except ValueError as e:
                    return _err(str(e))
            if sid in OPTIMISE_STUDIES and getattr(args, 'optimise_settings', None):
                from .core.optimise.search import read_settings
                try:
                    snap.analysis.studies[sid] = {**(snap.analysis.studies.get(sid) or {}),
                                                  **read_settings(args.optimise_settings)}
                except jobs.StudyError as e:
                    return _err(str(e))
            try:
                doc = jobs.run_study(sid, snap)
            except jobs.StudyError as e:
                from . import batch_track
                batch_track.record(sid, None, project if project is not None else snap, reason=str(e))
                return _err(str(e))
            except Exception as e:                      # noqa: BLE001 — a sentence here, the traceback in a file
                return _unexpected(jobs.STUDIES[sid].title, e)
            doc = {k: v for k, v in doc.items() if k != 'cached'}
            option = jobs.STUDIES[sid].batch.replace('-', '_')
            try:
                dest = study_dest(dest, jobs.STUDIES[sid].batch)
            except SystemExit as e:
                return int(e.code)
            doc = PV.stamp(doc, project if project is not None else snap,
                           output=PV.batch_output(doc, dest, option=option))
            _write(doc, dest)
            if dest != '-':
                print(f'{jobs.STUDIES[sid].title} → {dest}')
            from . import batch_track
            batch_track.record(sid, doc, project if project is not None else snap, dest)
            if sid == 'audit':
                code = max(code, _audit_exports(args, doc, project if project is not None else snap, snap))
                args._audit_doc = doc                   # for --rerun
        if getattr(args, 'synth_episodes', None):
            from .core import jobs
            from .core.sim.episodes import batch_generate
            if project is None:
                return _err('--synth-episodes needs a project file (or --sample)')
            try:
                print(batch_generate(project, args.synth_episodes, args.synth_settings))
            except jobs.StudyError as e:
                return _err(str(e))
        if getattr(args, 'experiment', None) or getattr(args, 'experiment_out', None):
            code = max(code, _experiment(args, project))
        if getattr(args, 'export', None) or getattr(args, 'formats', None) or getattr(args, 'export_options', None):
            code = max(code, _export(args, project))
        if getattr(args, 'rerun', None) or getattr(args, 'rerun_audit', None):
            from .core import jobs
            from .core.bridges import rerun as RB
            if not getattr(args, 'rerun', None):
                return _err('--rerun-audit goes with --rerun OUT.rrd')
            if project is None:
                return _err('--rerun needs a project file (or --sample)')
            try:
                print(RB.batch(project, args.rerun, getattr(args, 'rerun_audit', None),
                               getattr(args, '_audit_doc', None)))
            except jobs.StudyError as e:
                return _err(str(e))
        if args.validate:
            from . import validation as VA
            from .validation.suite import qt_app
            st = {'quick': bool(args.quick)}
            if args.models:
                known = VA.available_models()               # the plugins' models included
                bad = [m for m in args.models.split(',') if m.strip() not in known]
                if bad:
                    return _err(f'unknown model id(s) {", ".join(bad)}; known: {", ".join(known)}')
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
            if res.not_run:
                from .validation.suite import not_run_sentence
                print(not_run_sentence(res.not_run))
            code = max(code, VA.exit_code(res))
        from . import batch_gate
        code = max(code, batch_gate.run(args, project))      # --check, --run-plan
        from . import batch_research
        if batch_research.wanted(args):
            code = max(code, batch_research.run(args, project))     # --diff, --merge, --unpack, --cite, ...
        from . import batch_campaign
        if batch_campaign.wanted(args):
            code = max(code, batch_campaign.run(args, project))     # --campaign SPEC.json
        from . import batch_cells
        if batch_cells.wanted(args):
            code = max(code, batch_cells.run(args, project))        # --cells (batch_cells.py)
        from . import batch_stats
        if batch_stats.wanted(args):
            code = max(code, batch_stats.run(args, project))        # --stats SPEC.json (batch_stats.py)
        from . import batch_splat
        if batch_splat.wanted(args):
            code = max(code, batch_splat.run(args, project))        # --splat ACTION (batch_splat.py)
        from . import batch_track                                   # last: it reports the runs logged above
        if batch_track.wanted(args) or batch_track._FAILED:
            code = max(code, batch_track.run(args, project))        # --runs-export, runs not logged
        return code
    finally:
        if ctx is not None:
            PV.end_batch()


def _formats(args) -> list:
    """The --formats ids, aliases resolved (core/exporters ALIASES: 'usd' is the Isaac export)."""
    from .core import exporters as X
    return X.resolve([f.strip() for f in (getattr(args, 'formats', None) or '').split(',') if f.strip()])


def _captures(args) -> dict:
    """--captures DIR [--joints FILE]: the hand-eye solve's run-time capture settings ('folder', 'joints')."""
    folder = args.captures
    if not os.path.isdir(folder):
        raise ValueError(f'--captures {folder} is not a folder: give the folder of board images.')
    joints = getattr(args, 'joints', None)
    if joints is None:
        joints = next((os.path.join(folder, n) for n in JOINTS_FILES if os.path.isfile(os.path.join(folder, n))),
                      None)
        if joints is None:
            raise ValueError(f'The folder {os.path.basename(os.path.normpath(folder))} has no joint states file '
                             f'({", ".join(JOINTS_FILES)}): put one there or name it with --joints FILE.')
    elif not os.path.isfile(joints):
        raise ValueError(f'--joints {joints} is not a file.')
    return {'folder': folder, 'joints': joints}


def _experiment(args, project) -> int:
    """--experiment DESIGN.json [--experiment-out OUT.json]: core/experiments.py's batch run."""
    from .core import experiments as EX
    if not getattr(args, 'experiment', None):
        return _err('--experiment-out goes with --experiment DESIGN.json')
    if project is None:
        return _err('--experiment needs a project file (or --sample)')
    try:
        doc, jp, cp = EX.batch(project, args.experiment, getattr(args, 'experiment_out', None))
    except EX.ExperimentError as e:
        return _err(str(e))
    except Exception as e:                              # noqa: BLE001 — a sentence here, the traceback in a file
        return _unexpected('The experiment', e)
    h = doc.get('headline') or {}
    print(f'Designed experiment ({doc["method_label"]}, {doc["execution"]["runs_done"]} runs): '
          f'{h.get("word", "")} → {jp}, {cp}')
    from . import batch_track
    batch_track.record('experiments', doc, project, jp)
    return EXIT_OK


def _progress_line():
    """fn(fraction, text) for a terminal: a line on stderr each time the step's text changes (the report export runs
    every study, a minute or two with nothing else printed); not used when stderr is a file or a pipe."""
    last = ['']

    def say(frac, text=''):
        if text and text != last[0]:
            last[0] = text
            print(f'  {text} ({100 * frac:.0f} %)', file=sys.stderr, flush=True)
    return say


def _export(args, project) -> int:
    """--export OUT_DIR --formats a,b [--export-options JSON]: core.exporters.export. Exit 0 when every format was
    written, 1 when some were refused (the others are written; each refusal is said in a line), 2 when none could
    be (or a format name is unknown, refused in one line before anything is written)."""
    from .core import exporters as X
    from .core import jobs
    if not getattr(args, 'export', None) or not getattr(args, 'formats', None):
        return _err('--export OUT_DIR and --formats A,B,... go together')
    if project is None:
        return _err('--export needs a project file (or --sample)')
    X.load_exporters()
    fmts = _formats(args)
    unknown = [f for f in fmts if f not in X.EXPORTERS]
    if not fmts or unknown:
        return _err(f'unknown export format{"s" if len(unknown) > 1 else ""} {", ".join(unknown) or "(none given)"}; '
                    f'known: {", ".join(sorted(X.EXPORTERS))}')
    options = {}
    if getattr(args, 'export_options', None):
        text = args.export_options
        try:
            if not text.lstrip().startswith('{'):
                with open(text, encoding='utf-8') as fh:
                    text = fh.read()
            options = json.loads(text)
        except (OSError, ValueError) as e:
            return _err(f'--export-options could not be read as JSON ({e})')
        if not isinstance(options, dict) or any(not isinstance(v, dict) for v in options.values()):
            return _err('--export-options must map each format to its options, e.g. {"masks": {"atlas": false}}')
        options = {X.ALIASES.get(k, k): v for k, v in options.items()}
        bad = [k for k in options if k not in fmts]
        if bad:
            return _err(f'--export-options names {", ".join(bad)}, which --formats does not ask for')
    if 'fiftyone' in fmts and getattr(args, 'dataset', None):
        options.setdefault('fiftyone', {}).setdefault('dataset', args.dataset)
    ctx = jobs.Ctx(progress=_progress_line()) if sys.stderr.isatty() else None   # the report runs every study
    try:
        man = X.export(project.snapshot(), args.export, fmts, options, ctx=ctx)
    except jobs.StudyError as e:
        return _err(str(e))
    except Exception as e:                              # noqa: BLE001 — a sentence here, the traceback in a file
        return _unexpected('The export', e)
    ok = 0
    for fid, row in man['formats'].items():
        if row['status'] == 'ok':
            ok += 1
            print(f'{fid}: {len(row["files"])} file{"" if len(row["files"]) == 1 else "s"}')
            for n in row['notes']:
                print(f'  · {n}')
        else:
            print(f'{fid}: refused: {row.get("reason", "")}', file=sys.stderr)
    print(f'{ok} of {len(fmts)} formats written; manifest → {os.path.join(args.export, "export_manifest.json")}')
    return EXIT_OK if ok == len(fmts) else (EXIT_FAIL if ok else EXIT_ERROR)


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
    """--reproduce (batch_gate.py, core/reproduce.py): the project given, else the one the result names."""
    from . import batch_gate
    return batch_gate.reproduce(args, project)
