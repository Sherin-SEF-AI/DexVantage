"""Work: a list of independent items (an experiment's runs, a Monte Carlo's draws, an optimiser generation's designs,
any list of (project, study, settings) jobs), what each item computes, how the items are split into shards and how
their results are put back together into the result an in-process run gives.

    experiment_work(project, design)            the rig as it is, then every run of the design (core/experiments)
    tolerance_work(project, settings)           the nominal, then every draw (core/tolerance, core/studies/tolerance)
    studies_work([(project, study, settings)])  any studies on any projects
    designs_work(base, paths, rows, ...)        an optimiser generation's designs (core/optimise/rig)
    shard_indices(m, i, n)                      the items of shard i of n: i, i + n, i + 2n, ... (every item once)
    execute(work, index, state)                 one item's result, as JSON ('ok', result) or ('failed', sentence)
    assemble(work, results)                     the in-process result from every item's result

Work on disk (Work.save, Work.load): `work.json` (schema dexvantage.cluster-work/1: the kind, the items, the parameters,
each project's file name and content hash, the modules a worker imports, the work id) and the projects as .dexv files
beside it. The work id is the sha256 of the canonical JSON of the kind, the items, the parameters and the projects'
content hashes: shard outputs carry it, so outputs of another work are never mixed in.

Why the results are the in-process ones: each item is computed by the very function the in-process run calls on it
(core.experiments.evaluate_row, core.metrics.evaluate on core.tolerance.apply_draw's project, core.jobs.run_study,
the optimiser's evaluator), on the same project (the saved project's content hash is checked when it is loaded), and
the assembly calls the in-process code's own analysis (core.experiments._result; the tolerance study with an evaluator
that hands back the shards' figures in draw order). The draws are made again from the seed in every process, in the
in-process order, so draw i is the same everywhere.

Known limits: a study whose figure depends on the machine (the GPU and CPU occlusion paths can differ at a silhouette)
gives the same result only when every shard runs on the same kind of machine as the run it is compared with; a study
or metric a plugin adds must be installed where the shards run (the work names the modules to import); robot model
files are inside the .dexv, but a dataset a data study reads by path must be at the same path on every node.
"""
from __future__ import annotations

import copy
import json
import os
import time
import traceback
from dataclasses import dataclass, field

from .. import jobs

SCHEMA = 'dexvantage.cluster-work/1'
KINDS = ('experiment', 'tolerance', 'studies', 'designs')
KIND_TITLE = {'experiment': 'Designed experiment', 'tolerance': 'Monte Carlo tolerances',
              'studies': 'Study jobs', 'designs': 'Optimiser designs'}
ITEM_WORD = {'experiment': 'run', 'tolerance': 'draw', 'studies': 'job', 'designs': 'design'}


class ClusterError(jobs.StudyError):
    """Work that cannot be built, run or gathered; the message says why in plain words."""


def plain(o):
    """o as plain JSON values (a JSON round trip)."""
    return json.loads(json.dumps(o, allow_nan=True))


def shard_indices(m: int, i: int, n: int) -> list:
    """The item indices of shard i of n over m items: i, i + n, i + 2n, ... Over i = 0 … n − 1 every index in
    range(m) appears exactly once (a strided split, so slow and fast items late in a design spread over the shards)."""
    m, i, n = int(m), int(i), int(n)
    if n < 1:
        raise ClusterError(f'The number of shards must be at least 1; it is {n}.')
    if not 0 <= i < n:
        raise ClusterError(f'Shard {i} does not exist: the shards are numbered 0 to {n - 1}.')
    return list(range(i, max(m, 0), n))


def partition(m: int, n: int) -> list:
    """[shard_indices(m, i, n) for i in range(n)]."""
    return [shard_indices(m, i, n) for i in range(int(n))]


@dataclass
class Work:
    kind: str
    items: list
    params: dict
    projects: dict = field(default_factory=dict)        # name -> core.project.Project (in memory)
    modules: list = field(default_factory=list)         # imported by every worker (studies and metrics outside core)
    title: str = ''
    digests: dict = field(default_factory=dict)         # name -> the project's content hash
    folder: str = ''                                    # where it was loaded from or saved to

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ClusterError(f'Unknown kind of work "{self.kind}"; one of {", ".join(KINDS)}.')
        if not self.digests:
            from .. import provenance as PV
            self.digests = {k: PV.project_digest(p)['sha256'] for k, p in self.projects.items()}

    @property
    def work_id(self) -> str:
        from .. import provenance as PV
        return PV.sha256(PV.canonical({'kind': self.kind, 'items': self.items, 'params': self.params,
                                       'projects': self.digests}))[:32]

    def describe(self) -> str:
        n = len(self.items)
        w = ITEM_WORD[self.kind]
        t = self.title or KIND_TITLE[self.kind]
        if self.kind in ('experiment', 'tolerance'):      # item 0 is the project as it is
            first = 'the rig as it is' if self.kind == 'experiment' else 'the nominal rig'
            return f'{t}: {first} and {n - 1} {w}{"" if n == 2 else "s"}'
        return f'{t}: {n} {w}{"" if n == 1 else "s"}'

    def to_doc(self) -> dict:
        from ... import __version__
        return {'schema': SCHEMA, 'kind': self.kind, 'title': self.title, 'work_id': self.work_id,
                'dexvantage': __version__, 'items': self.items, 'params': self.params,
                'projects': {k: {'file': f'projects/{k}.dexv', 'sha256': self.digests[k]} for k in self.projects},
                'modules': list(self.modules)}

    def save(self, folder: str) -> str:
        """work.json and projects/<name>.dexv in folder; the work.json path."""
        from .cache import atomic_write
        os.makedirs(os.path.join(folder, 'projects'), exist_ok=True)
        for k, p in self.projects.items():
            p.snapshot().save(os.path.join(folder, 'projects', f'{k}.dexv'))
        path = os.path.join(folder, 'work.json')
        atomic_write(path, (json.dumps(self.to_doc(), indent=1, ensure_ascii=False, allow_nan=True) + '\n')
                     .encode('utf-8'))
        self.folder = folder
        return path

    @staticmethod
    def load(path: str) -> 'Work':
        """A Work from work.json (or the folder holding it); ClusterError in a sentence when it cannot be used."""
        from .. import provenance as PV
        from ..project import Project, ProjectFormatError
        if os.path.isdir(path):
            path = os.path.join(path, 'work.json')
        try:
            with open(path, encoding='utf-8') as fh:
                doc = json.load(fh)
        except OSError as e:
            raise ClusterError(f'The work file {os.path.basename(path)} could not be read: {e.strerror or e}.') \
                from None
        except ValueError as e:
            raise ClusterError(f'The work file {os.path.basename(path)} is not valid JSON ({e}).') from None
        if not isinstance(doc, dict) or doc.get('schema') != SCHEMA:
            raise ClusterError(f'{os.path.basename(path)} is not a DexVantage work file (schema {SCHEMA}).')
        folder = os.path.dirname(os.path.abspath(path))
        projects, digests = {}, {}
        for k, ref in (doc.get('projects') or {}).items():
            fp = os.path.join(folder, ref['file'])
            try:
                p = Project.load(fp)
            except ProjectFormatError as e:
                raise ClusterError(f'The work\'s project {ref["file"]} could not be read: {e}') from None
            got = PV.project_digest(p)['sha256']
            if got != ref.get('sha256'):
                raise ClusterError(f'The project {ref["file"]} changed since the work file was written (its content '
                                   'hash differs): write the job folder again.')
            projects[k], digests[k] = p, got
        w = Work(doc.get('kind'), doc.get('items') or [], doc.get('params') or {}, projects,
                 list(doc.get('modules') or []), str(doc.get('title') or ''), digests, folder)
        if doc.get('work_id') and doc['work_id'] != w.work_id:
            raise ClusterError('The work file was edited after it was written (its work id does not match its '
                               'content): write the job folder again.')
        return w


# ====================================================================== building work
def _modules_for(ids) -> list:
    from .. import experiments as EX
    return [m for m in EX._modules_for(ids) if not m.startswith('dexvantage.core.')]


def experiment_work(project, design) -> Work:
    """The rig as it is (item 0) and every run of the design (items 1 … n), as core.experiments.run evaluates them."""
    from .. import experiments as EX
    design = EX.load_design(design, project)
    problems = EX.check_design(project, design)
    if problems:
        raise EX.ExperimentError(' '.join(problems[:5]) + (f' ({len(problems) - 5} more.)' if len(problems) > 5
                                                           else ''))
    plan = EX.make_plan(design)
    rows = EX.plan_rows(plan, design.factors)
    base = project.snapshot()
    base.analysis.studies.pop(EX.STUDY_ID, None)          # as core.experiments.run: the design is no study's input
    ids = list(design.responses)
    items = [{'values': {}}] + [{'values': plain(r)} for r in rows]
    params = {'design': plain(design.to_dict()), 'variant': design.variant, 'ids': ids}
    return Work('experiment', items, params, {'base': base}, _modules_for(ids),
                design.name or f'{EX.METHOD_LABEL[plan.method]} ({plan.n} runs)')


def tolerance_settings(project, settings=None):
    """(project with the settings laid over the stored ones, the study's settings, the metric ids, where they came
    from)."""
    from ..studies import tolerance as ST
    p = project.snapshot()
    if settings:
        p.analysis.studies[ST.SID] = {**(p.analysis.studies.get(ST.SID) or {}), **dict(settings)}
    st = ST.settings_of(p)
    ids, why = ST.metric_ids(p, st)
    return p, st, ids, why


def tolerance_work(project, settings=None) -> Work:
    """The project as it is (item 0) and every draw of its mount tolerances (items 1 … n)."""
    from .. import tolerance as TL
    p, st, ids, _why = tolerance_settings(project, settings)
    if not TL.sources(p, st['scale'], st['distribution'], st['sensors'] or None):
        raise ClusterError('There is no enabled sensor in this rig to perturb: add one (Sensors) first.')
    items = [{'draw': None}] + [{'draw': i} for i in range(st['n'])]
    params = {'ids': ids, 'n': st['n'], 'seed': st['seed'], 'scale': st['scale'], 'distribution': st['distribution'],
              'sensors': st['sensors']}
    return Work('tolerance', items, params, {'base': p}, _modules_for(ids), f'Monte Carlo tolerances ({st["n"]} draws)')


def studies_work(job_list) -> Work:
    """Any studies on any projects: job_list of (project, study id, settings) or {'project', 'study', 'settings'}."""
    from .. import provenance as PV
    jobs.load_studies()
    projects, by_digest, items = {}, {}, []
    for k, j in enumerate(job_list):
        if isinstance(j, dict):
            p, sid, st = j.get('project'), j.get('study'), j.get('settings')
        else:
            p, sid, st = (list(j) + [None])[:3]
        p = getattr(p, 'project', p)                       # a dv.Rig
        if sid not in jobs.STUDIES:
            raise ClusterError(f'Job {k + 1}: there is no study called "{sid}".')
        if st is not None and not isinstance(st, dict):
            raise ClusterError(f'Job {k + 1}: the settings must be a JSON object.')
        d = PV.project_digest(p)['sha256']
        if d not in by_digest:
            by_digest[d] = f'p{len(projects)}'
            projects[by_digest[d]] = p.snapshot()
        items.append({'project': by_digest[d], 'study': sid, 'settings': plain(st or {})})
    if not items:
        raise ClusterError('There is no job to run.')
    mods = sorted({getattr(jobs.STUDIES[i['study']].fn, '__module__', '') for i in items} - {'', '__main__'})
    return Work('studies', items, {}, projects, [m for m in mods if not m.startswith('dexvantage.core.')],
                f'{len(items)} study job{"" if len(items) == 1 else "s"}')


def design_key(values) -> str:
    """The optimiser memo's key of a design (core/optimise/rig._key) as a JSON string."""
    from ..optimise import rig
    return json.dumps(plain([list(v) if isinstance(v, tuple) else v for v in rig._key(values)]))


def designs_work(base, paths, rows, objectives, constraints, screen) -> Work:
    """An optimiser generation's designs: each the factors at paths set to one row of values on base."""
    from .. import metrics as M
    items = [{'values': plain(list(v)), 'key': design_key(v)} for v in rows]
    ids = list(dict.fromkeys(list(objectives) + [c['metric'] for c in constraints]))
    M.load_metrics()
    params = {'paths': list(paths), 'objectives': list(objectives), 'constraints': plain(constraints),
              'screen': plain(screen or {})}
    return Work('designs', items, params, {'base': base}, _modules_for(ids),
                f'Optimiser designs ({len(items)})')


# ====================================================================== running one item
def prepare(work: Work) -> None:
    """Import the modules the work names and register every metric and study (a fresh process has none yet)."""
    import importlib
    from .. import metrics as M
    for mod in work.modules:
        try:
            importlib.import_module(mod)
        except Exception:                                  # noqa: BLE001 — its metrics then say "no metric called"
            traceback.print_exc()
    M.load_metrics()
    jobs.load_studies()


class State:
    """What a process keeps while it runs items of one work: the result cache and the draws."""

    def __init__(self, work: Work, cache: jobs.ResultCache | None = None):
        prepare(work)
        self.work = work
        self.cache = cache if cache is not None else jobs.ResultCache(8)
        self._draws = None
        self._ev = None

    def draws(self) -> list:
        """The tolerance draws, made again from the seed in core.tolerance.monte_carlo's order."""
        if self._draws is None:
            import numpy as np

            from .. import tolerance as TL
            w = self.work.params
            p = self.work.projects['base']
            srcs = TL.sources(p, w['scale'], w['distribution'], w['sensors'] or None)
            rng = np.random.default_rng(int(w['seed']))
            self._draws = [{s.uid: TL.draw_source(s, rng, w['scale']) for s in srcs} for _i in range(int(w['n']))]
        return self._draws

    def evaluator(self):
        if self._ev is None:
            from ..optimise import search as S
            w = self.work.params
            self._ev = S.Evaluator(w['objectives'], w['constraints'], w['screen'], cache=self.cache)
        return self._ev


def run_item(work: Work, item: dict, state: State, cancel=None):
    """The JSON-able result of one item (raises on an unexpected error; execute() turns that into a sentence)."""
    from .. import metrics as M
    k = work.kind
    ctx = jobs.Ctx(cancel=cancel)
    if k == 'experiment':
        from .. import experiments as EX
        w = work.params
        return EX.evaluate_row(work.projects['base'], item['values'], w['ids'], w['variant'], state.cache, cancel)
    if k == 'tolerance':
        from .. import tolerance as TL
        p = work.projects['base']
        if item['draw'] is not None:
            p = TL.apply_draw(p, state.draws()[int(item['draw'])])
        return M.evaluate(p, work.params['ids'], cache=state.cache, ctx=ctx)
    if k == 'studies':
        q = work.projects[item['project']].snapshot()
        sid = item['study']
        if item.get('settings'):
            q.analysis.studies[sid] = {**(q.analysis.studies.get(sid) or {}), **item['settings']}
        try:
            doc = jobs.run_study(sid, q, ctx, state.cache)
        except jobs.StudyError as e:
            return {'refused': str(e)}
        return {'doc': {kk: v for kk, v in doc.items() if kk != 'cached'}}
    if k == 'designs':
        from .. import factors as FA
        p = FA.apply(work.projects['base'], dict(zip(work.params['paths'], item['values'])))
        figs, status, reasons, bounds = state.evaluator()._figures(p)
        return {'figures': figs, 'status': status, 'reasons': reasons, 'bounds': bounds}
    raise ClusterError(f'Unknown kind of work "{k}".')


def execute(work: Work, index: int, state: State, cancel=None, item: dict | None = None) -> tuple:
    """('ok', result as plain JSON, seconds) or ('failed', a sentence, seconds) for item index (item: the item itself
    when it is not in work.items, as an optimiser generation's designs); ('cancelled', None, seconds) when cancel()
    turned true while it ran (an experiment run stopped part way gives no value to keep)."""
    t0 = time.perf_counter()
    try:
        res = run_item(work, work.items[index] if item is None else item, state, cancel)
    except Exception as e:                                 # noqa: BLE001 — a sentence in the output, the traceback logged
        traceback.print_exc()
        first = (str(e).splitlines() or [''])[0][:200]
        what = ITEM_WORD[work.kind]
        return ('failed', f'{what.capitalize()} {index} stopped with an unexpected error ({type(e).__name__}: '
                          f'{first}); the details are in the log.', round(time.perf_counter() - t0, 4))
    if res is None or (cancel is not None and cancel()):
        return 'cancelled', None, round(time.perf_counter() - t0, 4)
    return 'ok', plain(res), round(time.perf_counter() - t0, 4)


# ====================================================================== the in-process result from the items' results
def assemble(work: Work, results: dict, execution: dict | None = None) -> dict:
    """The result an in-process run gives, from {index: result} for every item (ClusterError when one is missing).
    execution: what the 'execution' block of an experiment says about how it ran."""
    missing = [i for i in range(len(work.items)) if i not in results]
    if missing:
        raise ClusterError(f'{len(missing)} of {len(work.items)} {ITEM_WORD[work.kind]}s have no result yet (the '
                           f'first is {missing[0]}): run the shards that hold them, then gather again.')
    execution = execution or {}
    prepare(work)
    if work.kind == 'experiment':
        return _assemble_experiment(work, results, execution)
    if work.kind == 'tolerance':
        return _assemble_tolerance(work, results, execution)
    if work.kind == 'studies':
        out = []
        for i, it in enumerate(work.items):
            r = results[i]
            out.append({'job': i + 1, 'study': it['study'], 'project': work.projects[it['project']].name,
                        'settings': it['settings'], **({'doc': r['doc']} if 'doc' in r else {'refused': r['refused']})})
        return {'schema': 'dexvantage.cluster-studies/1', 'jobs': out, 'execution': execution}
    return {'schema': 'dexvantage.cluster-designs/1',
            'designs': {work.items[i]['key']: results[i] for i in range(len(work.items))}, 'execution': execution}


def _assemble_experiment(work: Work, results: dict, execution: dict) -> dict:
    import numpy as np

    from .. import experiments as EX
    base = work.projects['base']
    design = EX.load_design(copy.deepcopy(work.params['design']), base)
    plan = EX.make_plan(design)
    runs = EX.plan_rows(plan, design.factors)
    if plain(runs) != [it['values'] for it in work.items[1:]]:
        raise ClusterError('This installation makes other runs from the design than the work file holds (another '
                           'version of DexVantage or SciPy?): gather with the version that wrote the job folder.')
    ids = list(work.params['ids'])
    base_res = results[0]
    Y = np.full((plan.n, len(ids)), np.nan)
    res: list = [None] * plan.n
    for i in range(plan.n):
        r = results[i + 1]
        res[i] = r
        Y[i] = [np.nan if r[mid]['value'] is None else r[mid]['value'] for mid in ids]
    ex = {'workers': int(execution.get('workers', 0)), 'mode': execution.get('mode', 'core/cluster'),
          'runs_done': plan.n, 'runs_planned': plan.n, 'cache_hits': None, 'runs_from_cache': None,
          'seconds': execution.get('seconds', 0.0), **{k: v for k, v in execution.items()
                                                      if k not in ('workers', 'mode', 'seconds')}}
    doc = EX._result(base, design, plan, runs, Y, res, base_res, ids, ex, False)
    return {'schema': 'dexvantage.experiments/1', 'study': EX.STUDY_ID, **doc}


def _assemble_tolerance(work: Work, results: dict, execution: dict) -> dict:
    from ..studies import tolerance as ST
    order = iter([copy.deepcopy(results[i]) for i in range(len(work.items))])

    def replay(_project, _ids, _ctx):
        return next(order)
    t0 = time.perf_counter()
    doc = ST.run(work.projects['base'], jobs.Ctx(), evaluator=replay)
    return {'schema': f'dexvantage.{ST.SID}/1', 'study': ST.SID, **doc,
            'seconds': execution.get('seconds', round(time.perf_counter() - t0, 4)), 'cancelled': False,
            'cached': False, 'execution': execution}
