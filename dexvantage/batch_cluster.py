"""Batch mode's cluster sweeps (core/cluster), added to batch.py's parser.

    dexvantage P.dexv --experiment DESIGN.json --jobs N        the experiment's runs in N processes on this machine
    dexvantage P.dexv --tolerance-mc OUT.json --jobs N         the tolerance draws in N processes
    dexvantage P.dexv --optimise OUT.json --jobs N             each optimiser generation's designs in N processes
    dexvantage --cluster-studies LIST.json --jobs N [--cluster-out OUT.json]
                                                               any (project, study, settings) jobs: LIST.json is
                                                               [{"project": "a.dexv", "study": "coverage",
                                                                 "settings": {...}}, ...] (paths beside LIST.json)
    ... --slurm FOLDER [--shards N] [--slurm-settings JSON] [--submit]
                                                               instead of running: a Slurm job folder (an array of
                                                               N shards; the optimiser: its first round)
    dexvantage --run-shard WORK.json --shard I --of N [--out DIR]
                                                               one shard (what each array task runs)
    dexvantage --gather FOLDER [--gather-out OUT.json]         merge the shards into the in-process result, or say
                                                               which shards are missing or failed and write
                                                               resubmit.sbatch for just those
    dexvantage --optimise-continue FOLDER [--submit]           the optimiser's next round (or its result)

--jobs 1 runs serially in this process; with no N the default is min(CPUs, available memory / 2 GiB per worker, a
labelled default). --cluster-cache DIR shares a result cache folder between the workers (core/cluster/cache.py).

Exit codes: 0 done, 1 a shard with failed items or a gather that is not complete (the report says which shards), 2 an
error that stopped the run (a sentence on stderr).
"""
from __future__ import annotations

import json
import os
import sys

EXIT_OK, EXIT_FAIL, EXIT_ERROR = 0, 1, 2
_ALL = ('run_shard', 'gather', 'jobs', 'slurm', 'cluster_studies', 'optimise_continue', 'shard', 'of', 'out',
        'gather_out', 'shards', 'slurm_settings', 'submit', 'cluster_out', 'cluster_cache')
TARGETS = ('experiment', 'study__experiments', 'study__tolerance', 'study__optimise', 'cluster_studies')


def add_arguments(g):
    from .core import provenance as PV
    g.add_argument('--jobs', type=int, metavar='N', help='with --experiment, --experiment-saved, --tolerance-mc, '
                   '--optimise or --cluster-studies: run in N worker processes on this machine (core/cluster)')
    g.add_argument('--slurm', metavar='FOLDER', help='with the same: write a Slurm job folder (an sbatch array, '
                   'docs/guide/cluster.md) instead of running')
    g.add_argument('--shards', type=int, metavar='N', help='with --slurm: the number of array tasks (default: the '
                   'items, at most 32); with --gather: the shard count when the folder does not record it')
    g.add_argument('--slurm-settings', metavar='JSON', help='with --slurm: sbatch settings as JSON text or a .json '
                   'file, e.g. {"mem": "8G", "time": "04:00:00", "partition": "cpu", "activate": "source '
                   '<venv>/bin/activate"}')
    g.add_argument('--submit', action='store_true', help='with --slurm or --optimise-continue: also submit with '
                   'sbatch')
    g.add_argument('--cluster-studies', metavar='LIST.json', help='run a list of (project, study, settings) jobs with '
                   '--jobs N or --slurm FOLDER')
    g.add_argument('--cluster-out', metavar='OUT.json', help='with --cluster-studies --jobs: the results file '
                   '(default LIST.results.json)')
    g.add_argument('--cluster-cache', metavar='DIR', help='with --jobs or --run-shard: a result cache folder shared '
                   'by the workers')
    g.add_argument('--run-shard', metavar='WORK.json', help='run one shard of a cluster work file')
    g.add_argument('--shard', type=int, metavar='I', help='with --run-shard: the shard (0 … N − 1)')
    g.add_argument('--of', type=int, metavar='N', help='with --run-shard: the number of shards')
    g.add_argument('--out', metavar='DIR', help='with --run-shard: the output folder (default: out/ beside the work)')
    g.add_argument('--gather', metavar='FOLDER', help='merge a job folder\'s shard outputs into the result')
    g.add_argument('--gather-out', metavar='OUT.json', help='with --gather: the result file (default '
                   'FOLDER/result.json)')
    g.add_argument('--optimise-continue', metavar='FOLDER', help='the next round of an optimiser job folder on Slurm')
    PV.INPUT_ARGS.setdefault('cluster_studies', 'first')


def wanted(args) -> bool:
    return any(getattr(args, k, None) is not None and getattr(args, k, None) is not False for k in _ALL)   # --jobs 0


def _err(text) -> int:
    print(f'dexvantage: {text}', file=sys.stderr)
    return EXIT_ERROR


def _json_arg(text, what):
    if not text.lstrip().startswith('{') and not text.lstrip().startswith('['):
        with open(text, encoding='utf-8') as fh:
            text = fh.read()
    v = json.loads(text)
    if not isinstance(v, dict):
        raise ValueError(f'{what} must be a JSON object')
    return v


def _targets(args) -> list:
    return [t for t in TARGETS if getattr(args, t, None)]


def run(args, project) -> int:
    from .core import jobs
    from .core import provenance as PV
    from .core.cluster import ClusterError
    parser = getattr(args, '_parser', None)
    ctx = PV.begin_batch(args, parser) if parser is not None else None
    try:
        if project is not None:
            PV.note_project(project)
        code = _run(args, project)
        from . import batch_track
        if batch_track.wanted(args) or batch_track._FAILED:
            code = max(code, batch_track.run(args, project))     # the tracker's summary line, runs not logged
        return code
    except (ClusterError, jobs.StudyError) as e:
        return _err(str(e))
    except Exception as e:                              # noqa: BLE001 — a sentence here, the traceback in a file
        from .batch import _unexpected
        return _unexpected('The cluster run', e)
    finally:
        if ctx is not None:
            PV.end_batch()


def _progress(f, text=''):
    print(f'\r{f * 100:5.1f} %  {text[:60]:60s}', end='', file=sys.stderr, flush=True)


def _run(args, project) -> int:  # noqa: C901 — one branch per command
    from .core import cluster as C
    modes = [k for k in ('run_shard', 'gather', 'optimise_continue') if getattr(args, k, None)]
    pool, slurm = args.jobs is not None, bool(args.slurm)
    if len(modes) + (pool or slurm) > 1 or (pool and slurm):
        return _err('give one of --run-shard, --gather, --optimise-continue, --jobs N or --slurm FOLDER')
    if (args.shard is not None or args.of is not None or args.out) and not args.run_shard:
        return _err('--shard, --of and --out go with --run-shard WORK.json')
    if args.gather_out and not args.gather:
        return _err('--gather-out goes with --gather FOLDER')
    if (args.slurm_settings or args.shards) and not (slurm or args.gather):
        return _err('--slurm-settings and --shards go with --slurm FOLDER')
    if args.submit and not (slurm or args.optimise_continue):
        return _err('--submit goes with --slurm FOLDER or --optimise-continue FOLDER')
    if args.cluster_out and not (args.cluster_studies and pool):
        return _err('--cluster-out goes with --cluster-studies LIST.json --jobs N')
    if args.cluster_cache and not (pool or args.run_shard):
        return _err('--cluster-cache goes with --jobs N or --run-shard')
    if args.run_shard:
        if args.shard is None or args.of is None:
            return _err('--run-shard needs --shard I and --of N')
        s = C.run_shard(args.run_shard, args.shard, args.of, args.out, cache_dir=args.cluster_cache)
        print(f'Shard {s["shard"]} of {s["of"]}: {s["done"]} done, {s["failed"]} failed, {s["skipped"]} already '
              f'done before; peak memory {s["peak_rss_mb"]:.0f} MB, {s["seconds"]:.1f} s → {s["file"]}')
        return EXIT_FAIL if s['failed'] else EXIT_OK
    if args.gather:
        rep = C.gather(args.gather, args.gather_out, shards=args.shards, project=project)
        print(rep['text'])
        if rep.get('result') is not None and rep['kind'] in ('experiment', 'tolerance'):
            _track('experiments' if rep['kind'] == 'experiment' else 'tolerance', rep['result'],
                   project if project is not None else _work_project(args.gather), rep.get('result_path'))
        for r in rep['rows']:
            if r['status'] != 'complete':
                print(f'  shard {r["shard"]}: {r["status"]}, {r["done"]} of {r["items"]} done'
                      + (f' ({r["error"]})' if r.get('error') else ''))
        return EXIT_OK if rep['complete'] else EXIT_FAIL
    if args.optimise_continue:
        r = C.optimise_slurm(args.optimise_continue, submit=args.submit)
        print(r['text'])
        return EXIT_OK
    if not (pool or slurm):
        return _err('--cluster-studies needs --jobs N or --slurm FOLDER') if args.cluster_studies else EXIT_OK
    tg = _targets(args)
    if len(tg) != 1:
        return _err('--jobs and --slurm go with one of --experiment DESIGN.json, --experiment-saved, --tolerance-mc, '
                    '--optimise OUT.json or --cluster-studies LIST.json')
    t = tg[0]
    if t != 'cluster_studies' and project is None:
        return _err('a project file (or --sample) is needed')
    if args.jobs is not None and args.jobs < 1:
        return _err('--jobs takes a number of processes, 1 or more')
    sl = {}
    if args.slurm_settings:
        try:
            sl = _json_arg(args.slurm_settings, '--slurm-settings')
        except (OSError, ValueError) as e:
            return _err(f'--slurm-settings could not be read as JSON ({e})')
    ctx = __import__('dexvantage.core.jobs', fromlist=['Ctx']).Ctx(progress=_progress if sys.stderr.isatty() else None)
    if t == 'study__optimise':
        st = {}
        if getattr(args, 'optimise_settings', None):
            from .core.optimise.search import read_settings
            st = read_settings(args.optimise_settings)
        if slurm:
            r = C.optimise_slurm(args.slurm, project, st, sl, args.shards, args.submit)
            print(r['text'])
            return EXIT_OK
        doc = C.optimise(project, st, args.jobs, ctx, args.cluster_cache)
        doc = {'schema': 'dexvantage.optimise/1', 'study': 'optimise', **doc}
        return _write_study(doc, project, args.study__optimise, 'optimise', 'optimise')
    work = _work(args, project, t)
    if slurm:
        job = C.slurm_folder(work, args.slurm, sl, args.shards)
        print(job['text'])
        if args.submit:
            from .core.cluster.slurm import submit
            print(f'Submitted array job {submit(args.slurm)}.')
        return EXIT_OK
    from .core.cluster.local import run_local
    doc = run_local(work, args.jobs, ctx, cache_dir=args.cluster_cache)
    if sys.stderr.isatty():
        print(file=sys.stderr)
    ex = doc.get('execution') or {}
    peak = ex.get('peak_rss_mb_per_worker')
    print(f'{work.describe()} on {ex.get("workers", 1)} process{"" if ex.get("workers", 1) == 1 else "es"} in '
          f'{ex.get("seconds", 0):.1f} s' + (f', peak {peak:.0f} MB per worker' if peak else '') + '.')
    if t == 'experiment':
        from .core import experiments as EX
        out = getattr(args, 'experiment_out', None) or EX.default_output(args.experiment)
        jp, cp = EX.write_outputs(doc, project, out)
        print(f'Designed experiment: {(doc.get("headline") or {}).get("word", "")} → {jp}, {cp}')
        _track('experiments', doc, project, jp)
        return EXIT_OK
    if t == 'study__experiments':
        return _write_study(doc, project, args.study__experiments, 'experiment_saved', 'experiments')
    if t == 'study__tolerance':
        return _write_study(doc, project, args.study__tolerance, 'tolerance_mc', 'tolerance')
    out = args.cluster_out or (args.cluster_studies[:-5] if args.cluster_studies.lower().endswith('.json') else
                               args.cluster_studies) + '.results.json'
    from .core.cluster.shards import write_result
    write_result(doc, project if project is not None else work.projects['p0'], out, 'studies',
                 option='cluster_studies')
    print(f'{len(doc["jobs"])} jobs → {out}')
    for k, r in enumerate(doc['jobs']):
        pj = work.projects.get(work.items[k].get('project')) if k < len(work.items) else None
        if pj is not None and isinstance(r.get('doc'), dict):
            _track(r['study'], r['doc'], pj, out)
    return EXIT_OK


def _track(sid: str, doc: dict, project, dest=None) -> None:
    """Log a cluster run in the experiment tracker when --track or the project's tracking asks for it
    (batch_track.record: never raises; a store that cannot be written is said at the end)."""
    if project is None or not isinstance(doc, dict):
        return
    from . import batch_track
    batch_track.record(sid, doc, project, dest)


def _work_project(folder: str):
    """The base project a job folder's work names (for logging a gathered result), or None."""
    try:
        from .core.cluster.work import Work
        w = Work.load(folder)
        return w.projects.get('base') or next(iter(w.projects.values()), None)
    except Exception:                                   # noqa: BLE001 — logging is a courtesy here
        return None


def _work(args, project, t):
    from .core import cluster as C
    from .core import experiments as EX
    if t == 'experiment':
        return C.experiment_work(project, EX.load_design(args.experiment, project))
    if t == 'study__experiments':
        st = EX.settings_of(project)
        if not st.get('factors'):
            raise EX.ExperimentError('The experiment has no factor yet: add one on Analysis ▸ Designed experiments, or '
                                     'run a design file with --experiment DESIGN.json.')
        return C.experiment_work(project, st)
    if t == 'study__tolerance':
        return C.tolerance_work(project)
    return C.studies_work(read_job_list(args.cluster_studies))


def read_job_list(path: str) -> list:
    """[(Project, study, settings)] from a jobs list file; project paths are relative to the file."""
    from .core.cluster import ClusterError
    from .core.project import Project, ProjectFormatError
    try:
        with open(path, encoding='utf-8') as fh:
            doc = json.load(fh)
    except OSError as e:
        raise ClusterError(f'The jobs list {os.path.basename(path)} could not be read: {e.strerror or e}.') from None
    except ValueError as e:
        raise ClusterError(f'The jobs list {os.path.basename(path)} is not valid JSON ({e}).') from None
    rows = doc.get('jobs') if isinstance(doc, dict) else doc
    if not isinstance(rows, list) or not rows:
        raise ClusterError('The jobs list must be a JSON list of {"project", "study", "settings"} objects.')
    base = os.path.dirname(os.path.abspath(path))
    loaded, out = {}, []
    for k, r in enumerate(rows):
        if not isinstance(r, dict) or not r.get('project') or not r.get('study'):
            raise ClusterError(f'Job {k + 1} of the list needs a "project" (a .dexv path) and a "study".')
        fp = r['project'] if os.path.isabs(r['project']) else os.path.join(base, r['project'])
        if fp not in loaded:
            try:
                loaded[fp] = Project.load(fp)
            except ProjectFormatError as e:
                raise ClusterError(f'Job {k + 1}: {e}') from None
        out.append((loaded[fp], r['study'], r.get('settings') or {}))
    return out


def _write_study(doc, project, dest, option, sid: str | None = None) -> int:
    from .batch import _write, study_dest
    from .core import provenance as PV
    doc = {k: v for k, v in doc.items() if k != 'cached'}
    try:
        dest = study_dest(dest, option.replace('_', '-'))
    except SystemExit as e:
        return int(e.code)
    doc = PV.stamp(doc, project, output=PV.batch_output(doc, dest, option=option))
    try:
        _write(doc, dest)
    except SystemExit as e:
        return int(e.code)
    if dest != '-':
        h = doc.get('headline') or {}
        print(f'{h.get("word", "done")} → {dest}')
    if sid:
        _track(sid, doc, project, dest)
    return EXIT_OK
