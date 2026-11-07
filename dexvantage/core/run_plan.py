"""Run a plan of studies, scorecard checks and exports in order, with their settings, and write one JSON summary
(spec §11: batch --run-plan plan.json [OUT_DIR]).

A plan is a JSON (or YAML) object {'name', 'steps': [...]} or a bare list of steps. Each step is one of

    {"study": "time_sync", "settings": {"skew_limit_ms": 1}, "out": "sync.json"}
    {"check": "requirements.yaml", "settings": {"grasp_visibility": {...}}, "out": "scorecard.json"}
                                          (a file next to the plan, "defaults", or a list of requirement items)
    {"export": "export/", "formats": ["mjcf", "opencv_yaml"], "options": {...}}
    {"reproduce": "earlier.json", "tolerance": ["1e-9"]}

Settings apply to that step only (merged into a snapshot's project.analysis.studies[study]; a check's settings are
{study: {...}}); the project itself is never changed. Outputs go under OUT_DIR (default: the plan's folder); each
result carries its provenance with the plan and the step (provenance 'plan'). A step that cannot run is recorded with
its sentence and the plan carries on.

Exit code (as --check): 2 when a step could not run, else 1 when a check fails or a reproduction does not match, else
0. The summary (schema dexvantage.run_plan/1) lists every step: what ran, its status, its output and result SHA-256,
the headline metrics of a study (core/metric_defs.py) and the verdict of a check.

Known limits: steps run one after another in this process (no parallelism); settings are per step, never carried to
the next step; a plan cannot loop or branch.
"""
from __future__ import annotations

import copy
import json
import os
import time

from . import jobs
from . import metrics as M
from . import provenance as PV
from .jobs import StudyError

SCHEMA = 'dexvantage.run_plan/1'
KINDS = ('study', 'check', 'export', 'reproduce')
EXIT_OK, EXIT_FAIL, EXIT_ERROR = 0, 1, 2


class PlanError(StudyError):
    """A plan that cannot be read or a step that names nothing it can run; the message says why."""


def load(path: str) -> dict:
    import yaml
    base = os.path.basename(path)
    try:
        with open(path, encoding='utf-8') as fh:
            doc = yaml.safe_load(fh.read())
    except OSError as e:
        raise PlanError(f'{base} could not be read: {e.strerror or e}.') from None
    except yaml.YAMLError as e:
        raise PlanError(f'{base} is not valid JSON or YAML: {getattr(e, "problem", None) or e}.') from None
    return normalise(doc, base)


def normalise(doc, name: str = 'The plan') -> dict:
    if isinstance(doc, list):
        doc = {'steps': doc}
    if not isinstance(doc, dict) or not isinstance(doc.get('steps'), list):
        raise PlanError(f'{name} is not a plan: a list of steps, or an object with "steps".')
    if not doc['steps']:
        raise PlanError(f'{name} has no steps.')
    steps = []
    for i, st in enumerate(doc['steps']):
        if not isinstance(st, dict):
            raise PlanError(f'Step {i + 1} of {name} is not an object.')
        kinds = [k for k in KINDS if k in st]
        if len(kinds) != 1:
            raise PlanError(f'Step {i + 1} of {name} must name exactly one of {", ".join(KINDS)}.')
        steps.append(dict(st, kind=kinds[0]))
    return {'name': str(doc.get('name') or os.path.splitext(name)[0]), 'steps': steps,
            'summary': doc.get('summary')}


def _headline_metrics(sid: str, doc: dict) -> dict:
    M.load_metrics()
    out = {}
    for m in M.METRICS.values():
        if m.study == sid and 'headline' in m.tags:
            r = M.read(m, doc)
            out[m.id] = {'value': r['value'], 'unit': m.unit, **({'bound': r['bound']} if r.get('bound') else {})}
    return out


def _out_path(out_dir: str, rel) -> str | None:
    if not rel:
        return None
    rel = str(rel)
    return rel if os.path.isabs(rel) else os.path.join(out_dir, rel)


def _write_json(doc, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    tmp = path + '.part'
    with open(tmp, 'w', encoding='utf-8') as fh:
        fh.write(json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(tmp, path)


def step_settings(step: dict) -> dict:
    """{study: settings} a step applies (a study step's settings are its study's)."""
    st = step.get('settings') or {}
    if step.get('kind', 'study' if 'study' in step else '') == 'study':
        return {str(step['study']): st}
    return st


def _snapshot(project, settings):
    snap = project.snapshot()
    for sid, st in (settings or {}).items():
        if not isinstance(st, dict):
            raise PlanError(f'The settings of {sid} are not an object.')
        snap.analysis.studies[sid] = {**(snap.analysis.studies.get(sid) or {}), **copy.deepcopy(st)}
    return snap


def run(project, plan: dict, out_dir: str, plan_dir: str | None = None, dataset: str | None = None,
        progress=None) -> tuple:
    """(summary doc, exit code). project: the plan's project (never changed). plan: normalise()d."""
    from . import requirements as RQ
    from . import reproduce as RP
    jobs.load_studies()
    plan_dir = plan_dir or out_dir
    b = PV.current_batch()
    pid = {'plan': plan['name'][:32], 'version': 1}
    rows, code = [], EXIT_OK
    for i, st in enumerate(plan['steps']):
        kind = st['kind']
        row = {'index': i + 1, 'kind': kind, 'what': st[kind] if not isinstance(st[kind], list) else 'items',
               'status': 'done', 'exit': EXIT_OK, 'out': st.get('out')}
        if progress:
            progress(i / len(plan['steps']), f'Step {i + 1}: {kind} {row["what"]}')
        if b is not None:
            b['plan'] = dict(pid, step=i)
        t0 = time.perf_counter()
        try:
            if kind == 'study':
                sid = str(st['study'])
                if sid not in jobs.STUDIES:
                    raise PlanError(f'There is no study "{sid}" (the studies are: {", ".join(sorted(jobs.STUDIES))}).')
                snap = _snapshot(project, step_settings(st))
                if dataset and sid in jobs.DATASET_STUDIES:
                    from .studies.audit import use_dataset
                    use_dataset(snap, dataset, sid)
                doc = {k: v for k, v in jobs.run_study(sid, snap).items() if k != 'cached'}
                out = _out_path(out_dir, st.get('out') or f'step{i + 1:02d}-{sid}.json')
                doc = PV.stamp(doc, snap, output={'option': '--run-plan', 'name': os.path.basename(out),
                                                  'in_folder': True}, plan=dict(pid, step=i))
                _write_json(doc, out)
                row.update(out=os.path.relpath(out, out_dir), study=sid,
                           result_sha256=doc[PV.KEY]['result_sha256'], metrics=_headline_metrics(sid, doc))
            elif kind == 'check':
                src = st['check']
                reqs = RQ.load(src if src == 'defaults' or os.path.isabs(str(src)) else
                               os.path.join(plan_dir, str(src))) if not isinstance(src, list) else \
                    [RQ.clean(x, f'Requirement {k + 1} of step {i + 1}') for k, x in enumerate(src)]
                snap = _snapshot(project, step_settings(st))
                card = RQ.check(snap, reqs)
                doc = {'schema': RQ.SCHEMA, 'study': 'requirements', **card}
                out = _out_path(out_dir, st.get('out') or f'step{i + 1:02d}-scorecard.json')
                doc = PV.stamp(doc, snap, output={'option': '--run-plan', 'name': os.path.basename(out),
                                                  'in_folder': True}, plan=dict(pid, step=i))
                _write_json(doc, out)
                ec = RQ.exit_code(card)
                row.update(out=os.path.relpath(out, out_dir), verdict=card['verdict'],
                           verdict_text=card['verdict_text'], counts=card['counts'],
                           result_sha256=doc[PV.KEY]['result_sha256'],
                           status={0: 'pass', 1: 'fail', 2: 'error'}[ec], exit=ec)
            elif kind == 'export':
                from .exporters import export
                dest = _out_path(out_dir, st['export'])
                fm = st.get('formats') or []
                fm = [f.strip() for f in fm.split(',')] if isinstance(fm, str) else list(fm)
                if not fm:
                    raise PlanError(f'Step {i + 1} exports no format: give "formats".')
                man = export(project, dest, fm, st.get('options') or {})
                refused = {f: r.get('reason') for f, r in (man.get('formats') or {}).items()
                           if r.get('status') == 'refused'}
                row.update(out=os.path.relpath(dest, out_dir), formats=fm, files=len(man.get('files') or {}),
                           refused=refused)
                if refused:
                    row.update(status='error', exit=EXIT_ERROR,
                               message=' '.join(f'{f}: {r}' for f, r in refused.items()))
            else:
                path = st['reproduce']
                path = path if os.path.isabs(str(path)) else os.path.join(plan_dir, str(path))
                tol = PV.parse_tolerances(st.get('tolerance') or [])
                rep = RP.reproduce_file(path, project, tol, dataset)
                ec = RP.exit_code(rep)
                row.update(verdict=rep['verdict'], message=rep.get('message', ''),
                           mismatches=len(rep.get('mismatches') or []),
                           status={0: 'reproduces', 1: 'does not reproduce', 2: 'error'}[ec], exit=ec)
        except (StudyError, ValueError, OSError) as e:
            msg = str(e) if not isinstance(e, OSError) else f'{e.strerror or e}'
            row.update(status='error', exit=EXIT_ERROR, message=msg)
        row['seconds'] = round(time.perf_counter() - t0, 3)
        code = max(code, row['exit'])
        rows.append(row)
    if b is not None:
        b['plan'] = dict(pid, step=-1)
    verdict = {EXIT_OK: 'done', EXIT_FAIL: 'a check failed', EXIT_ERROR: 'a step could not run'}[code]
    summary = {'schema': SCHEMA, 'plan': {'name': plan['name'], 'steps': [{k: v for k, v in s.items()}
                                                                          for s in plan['steps']]},
               'steps': rows, 'exit': code, 'verdict': verdict,
               'counts': {'steps': len(rows), 'errors': sum(r['exit'] == EXIT_ERROR for r in rows),
                          'failed': sum(r['exit'] == EXIT_FAIL for r in rows)}}
    return summary, code
