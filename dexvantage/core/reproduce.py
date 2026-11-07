"""Re-run what wrote a result and compare every number (spec §11: batch --reproduce result.json).

The result's provenance block names the study (or the scorecard, or the plan) that wrote it and the project it was
made from. reproduce() finds the project (the one given; else the sample its command names, --sample NAME; else the
project file it names, looked for next to the result and in the current folder), re-runs the study with the settings
the project holds, and compares every number and text of the two results (core/provenance.py compare: provenance,
time stamps and computation times left out, uids compared by place), each number exactly or within the stated
tolerances. A scorecard (--check, --scorecard) is re-run with the requirements recorded in it; a run-plan summary
(--run-plan) by reproducing each step's result file found next to it.

Computation-time figures that are not under a "seconds" key (COMPUTE_TIME: a study's timings, throughput and timing
blocks) are allowed to differ and are reported as such: they are measurements of this machine, not results.

Exit codes (as --check): 0 reproduces (exactly or within the tolerances), 1 does not reproduce, 2 cannot be tried
(no project, a study this build does not have, a file that cannot be read).

Known limits: a re-run uses the project's settings as they are now (they are in the project's content hash, so a
change is reported as "the project differs"); studies that read a recorded dataset need --dataset as when they ran;
GPU renders may differ in their last bits between drivers (give a tolerance).
"""
from __future__ import annotations

import json
import os

from . import jobs
from . import provenance as PV

REPORT_SCHEMA = PV.REPRODUCE_SCHEMA
BIG = 1e300                          # "any difference": computation times (a finite number: JSON has no infinity)
COMPUTE_TIME = ('*timings*', '*throughput*', 'timing', 'timing.*', '*.timing.*', 'timing_s')
EXIT_OK, EXIT_FAIL, EXIT_ERROR = 0, 1, 2


def _stem(doc) -> str:
    return PV._schema_stem(doc)


def _sample_of(prov) -> str | None:
    cmd = prov.get('command') or []
    for i, a in enumerate(cmd):
        if a == '--sample' and i + 1 < len(cmd):
            return str(cmd[i + 1])
    return None


def find_project(recorded: dict, result_path: str | None = None):
    """(project, how) for a result whose project was not given: its --sample, else its project file next to the
    result or in the current folder; (None, why) when there is none."""
    from .project import Project, ProjectFormatError
    prov = recorded.get(PV.KEY) or {}
    smp = _sample_of(prov)
    if smp:
        from .robot.samples import SampleError, sample_project
        try:
            return sample_project(smp), f'the {smp} sample its command names'
        except SampleError as e:
            return None, str(e)
    name = (prov.get('project') or {}).get('file')
    if not name:
        return None, 'the result names no project file: give the project it was made from'
    folders = [os.path.dirname(os.path.abspath(result_path))] if result_path else []
    folders.append(os.getcwd())
    for f in folders:
        path = os.path.join(f, os.path.basename(str(name)))
        if os.path.isfile(path):
            try:
                return Project.load(path), f'{os.path.basename(path)} next to the result' if f != os.getcwd() \
                    else f'{os.path.basename(path)} in the current folder'
            except ProjectFormatError as e:
                return None, str(e)
    return None, f'the project file it names ({os.path.basename(str(name))}) is not next to the result: give it'


def tolerances_with_defaults(tol) -> list:
    """The user's tolerances first (the first matching pattern applies), then the computation-time patterns."""
    return list(tol or []) + [(p, BIG, False) for p in COMPUTE_TIME]


def rerun(recorded: dict, project, dataset: str | None = None) -> dict:
    """The result made again the way the recorded one was made (unstamped); StudyError when it cannot be."""
    from . import requirements as RQ
    prov = recorded.get(PV.KEY) or {}
    stem = _stem(recorded)
    study = recorded.get('study') or prov.get('study')
    jobs.load_studies()
    snap = project.snapshot()
    if stem == 'requirements' and isinstance(recorded.get('requirements'), list):
        import time
        t0 = time.perf_counter()
        # the variant the scorecard was made for (a scorecard can be of any variant; --check scores the active one);
        # a project whose uids differ (a sample made again) has no such variant: its active one is scored
        vu = recorded.get('variant_uid')
        variant = vu if vu and snap.variant_by_uid(vu) is not None else None
        card = RQ.check(snap, [RQ.Requirement.from_dict(r) for r in recorded['requirements']], variant=variant)
        doc = {'schema': RQ.SCHEMA, 'study': 'requirements', **card}
        if 'seconds' in recorded:
            doc['seconds'] = round(time.perf_counter() - t0, 4)
        if 'cancelled' in recorded:
            doc['cancelled'] = False
        return doc
    if study == 'summary':
        doc = jobs.run_study('summary', snap)
    elif study in jobs.STUDIES:
        if dataset and study in jobs.DATASET_STUDIES:
            from .studies.audit import use_dataset
            use_dataset(snap, dataset, study)
        doc = jobs.run_study(study, snap)
    elif not (study or stem):
        raise jobs.StudyError('the file names no study (a result DexVantage writes carries "study" and a provenance '
                              'block), so there is nothing to run again')
    else:
        raise jobs.StudyError(f'this build cannot re-run "{study or stem}"')
    return {k: v for k, v in doc.items() if k != 'cached'}


def reproduce_doc(recorded: dict, name: str, project=None, tolerances=(), result_path: str | None = None,
                  dataset: str | None = None) -> dict:
    """The reproduction report (schema dexvantage.reproduce/1) of one result, with 'how' (where the project came
    from) and 'mismatches' (every number that differs, is missing or new, and every text that changed)."""
    tol = tolerances_with_defaults(tolerances)
    how = 'the project given'
    if _stem(recorded) == 'run_plan':
        return reproduce_plan(recorded, name, project, tolerances, result_path, dataset)
    if project is None:
        project, how = find_project(recorded, result_path)
        if project is None:
            rep = PV.report(name, recorded, None, tol, error=f'Cannot reproduce: {how}.')
            return dict(rep, how=None, mismatches=[])
    try:
        now = rerun(recorded, project, dataset)
    except jobs.StudyError as e:
        rep = PV.report(name, recorded, None, tol, error=f'Cannot reproduce: {e}.')
        return dict(rep, how=how, mismatches=[])
    now = PV.stamp(now, project)
    rep = PV.report(name, recorded, now, tol, now_provenance=now[PV.KEY], project_now=PV.project_digest(project))
    mism = [{'path': r[0], 'recorded': r[1], 'reproduced': r[2], 'status': r[3], 'difference': r[4],
             'tolerance': r[5]} for r in rep.get('numbers') or [] if r[3] in ('differs', 'missing', 'extra')]
    mism += [{'path': t[0], 'recorded': t[1], 'reproduced': t[2], 'status': 'text changed'}
             for t in rep.get('texts') or []]
    allowed = [r for r in rep.get('numbers') or [] if r[3] == 'within' and PV.tolerance_for(r[0], tol)[0] == BIG]
    return dict(rep, how=how, mismatches=mism, computation_times=len(allowed))


def reproduce_plan(summary: dict, name: str, project=None, tolerances=(), result_path: str | None = None,
                   dataset: str | None = None) -> dict:
    """A run-plan summary: every step's result file next to the summary reproduced on its own."""
    folder = os.path.dirname(os.path.abspath(result_path)) if result_path else os.getcwd()
    steps = []
    from . import run_plan as RN
    planned = (summary.get('plan') or {}).get('steps') or []
    if project is None:
        project, how = find_project(summary, result_path)
        if project is None:
            return {'schema': REPORT_SCHEMA, 'result': name, 'study': 'run_plan', 'verdict': 'cannot_reproduce',
                    'verdict_text': PV.VERDICT_TEXT['cannot_reproduce'], 'message': f'Cannot reproduce: {how}.',
                    'steps': [], 'mismatches': [], 'reasons': [how]}
    for st in summary.get('steps') or []:
        out = st.get('out')
        if not out or st.get('kind') not in ('study', 'check'):
            continue
        k = int(st.get('index') or 0) - 1
        step_project = project
        if 0 <= k < len(planned) and isinstance(planned[k], dict):
            try:
                step_project = RN._snapshot(project, RN.step_settings(planned[k]))
            except RN.PlanError:
                step_project = project
        path = os.path.join(folder, out)
        try:
            with open(path, encoding='utf-8') as fh:
                doc = json.load(fh)
        except (OSError, ValueError) as e:
            steps.append({'step': st.get('index'), 'out': out, 'verdict': 'cannot_reproduce',
                          'message': f'{out} could not be read: {getattr(e, "strerror", None) or e}.'})
            continue
        r = reproduce_doc(doc, out, step_project, tolerances, path, dataset)
        steps.append({'step': st.get('index'), 'out': out, 'verdict': r['verdict'], 'message': r.get('message', ''),
                      'mismatches': r.get('mismatches', [])[:50]})
    verdicts = [s['verdict'] for s in steps]
    if not steps:
        verdict, msg = 'cannot_reproduce', 'The plan summary names no step result to reproduce.'
    elif 'cannot_reproduce' in verdicts:
        verdict, msg = 'cannot_reproduce', 'A step result cannot be reproduced.'
    elif 'does_not_reproduce' in verdicts:
        verdict = 'does_not_reproduce'
        msg = f'Does not reproduce: {verdicts.count("does_not_reproduce")} of {len(steps)} step results differ.'
    elif 'reproduces_within_tolerance' in verdicts:
        verdict, msg = 'reproduces_within_tolerance', f'All {len(steps)} step results reproduce within the tolerances.'
    else:
        verdict, msg = 'reproduces', f'All {len(steps)} step results reproduce exactly.'
    return {'schema': REPORT_SCHEMA, 'result': name, 'study': 'run_plan', 'verdict': verdict,
            'verdict_text': PV.VERDICT_TEXT[verdict], 'message': msg, 'steps': steps,
            'mismatches': [dict(m, step=s['step']) for s in steps for m in s.get('mismatches') or []],
            'reasons': [f'step {s["step"]}: {s["message"]}' for s in steps if s['verdict'] not in
                        ('reproduces', 'reproduces_within_tolerance')]}


def reproduce_file(path: str, project=None, tolerances=(), dataset: str | None = None) -> dict:
    try:
        with open(path, encoding='utf-8') as fh:
            recorded = json.load(fh)
    except (OSError, ValueError) as e:
        return {'schema': REPORT_SCHEMA, 'result': os.path.basename(path), 'verdict': 'cannot_reproduce',
                'verdict_text': PV.VERDICT_TEXT['cannot_reproduce'], 'mismatches': [], 'reasons': [],
                'message': f'{os.path.basename(path)} could not be read: {getattr(e, "strerror", None) or e}.'}
    if not isinstance(recorded, dict):
        return {'schema': REPORT_SCHEMA, 'result': os.path.basename(path), 'verdict': 'cannot_reproduce',
                'verdict_text': PV.VERDICT_TEXT['cannot_reproduce'], 'mismatches': [], 'reasons': [],
                'message': f'{os.path.basename(path)} is not a DexVantage result (not a JSON object).'}
    return reproduce_doc(recorded, os.path.basename(path), project, tolerances, path, dataset)


def exit_code(rep: dict) -> int:
    v = rep.get('verdict')
    if v in ('reproduces', 'reproduces_within_tolerance'):
        return EXIT_OK
    return EXIT_ERROR if v == 'cannot_reproduce' else EXIT_FAIL


def lines(rep: dict, limit: int = 20) -> list:
    """The report as printed lines: the verdict, the reasons, then the mismatches (up to limit)."""
    out = [rep.get('message') or rep.get('verdict_text', '')]
    if rep.get('how'):
        out.append(f'  project: {rep["how"]}')
    c = rep.get('counts') or {}
    if c:
        out.append(f'  numbers: {c.get("numbers", 0)} compared, {c.get("exact", 0)} the same, {c.get("within", 0)} '
                   f'within tolerance ({rep.get("computation_times", 0)} computation times), {c.get("differs", 0)} '
                   f'differ; texts changed: {c.get("texts_changed", 0)}')
    for r in rep.get('reasons') or []:
        if r != out[0]:                                     # the message often is the only reason
            out.append(f'  · {r}')
    mm = rep.get('mismatches') or []
    for m in mm[:limit]:
        where = f'step {m["step"]}: ' if 'step' in m else ''
        out.append(f'  mismatch {where}{m["path"]}: {m.get("recorded")!r} then, {m.get("reproduced")!r} now'
                   f' ({m["status"]})')
    if len(mm) > limit:
        out.append(f'  … and {len(mm) - limit} more (in the report JSON)')
    return out
