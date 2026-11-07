"""Checks of the requirements scorecard, the CI gate, the metric definitions and --reproduce (core/requirements.py,
core/metric_defs.py, core/reproduce.py, batch_gate.py) against independent references.

- requirements.gate_exit_codes: batch --check on synthetic pass, fail, not-evaluated and error cases (a project and
  requirement files built here) against the exit code each case must give by spec §11 (0 pass, 1 fail, 2 error),
  written out per case. Tolerance 0 cases: an exit code is right or wrong.
- requirements.bound_decisions: the scorecard's decision on a figure stated only as a bound against a Monte Carlo
  over the true figures that bound allows: pass when every sampled true figure passes the comparison (written out
  again here, not the module's), fail when every one fails, else not evaluated. 4000 seeded random cases over the
  five comparisons, both bounds and no bound. Tolerance 0 disagreements: the rule is exact.
- requirements.metrics_vs_results: every metric of the cheap studies on the UR5e sample (with the bandwidth study's
  suggested host) through core/metrics.evaluate against the same figure read off the study's result by hand here.
  Tolerance 0 (the same number, nothing computed in between).
- requirements.reproduce_round_trip: two studies (bandwidth, time sync) on the UR5e sample, written with provenance
  and re-run by core/reproduce.py from the recorded result alone: numbers that differ, are missing or new, and texts
  that changed. Tolerance 0: the studies are deterministic.
- requirements.reproduce_detects_perturbation: one number of each recorded result changed by one part in a million
  and the result hash written again (an honest-looking file), then reproduced: the error counts the results whose
  mismatch list is not exactly that number. Tolerance 0.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from .suite import Outcome, check

OPS = ('>=', '<=', '<', '>', '==')


def _passes(true, op, t):
    """The comparison written out again (an independent copy of the definition, with the same 1e-9 relative slack)."""
    eps = 1e-9 * max(1.0, abs(t))
    return {'>=': true >= t - eps, '>': true > t + eps, '<=': true <= t + eps, '<': true < t - eps,
            '==': abs(true - t) <= eps}[op]


# ====================================================================== the gate
def _gate_cases(folder):
    import yaml

    from ..core.project import Sensor, blank_project
    p = blank_project('Gate cell')
    p.variant.sensors.append(Sensor(name='Wrist', kind='depth', model_id='intel-realsense-d405', slot=0))
    p.variant.sensors.append(Sensor(name='Top', kind='camera', model_id='', slot=1))
    proj = os.path.join(folder, 'gate.dexv')
    p.save(proj)

    def f(name, items=None, text=None):
        path = os.path.join(folder, name)
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(text if text is not None else yaml.safe_dump({'requirements': items}))
        return path
    return proj, [
        ('pass', f('pass.yaml', [{'metric': 'rig.camera_count', 'comparison': '<=', 'threshold': 2},
                                 {'metric': 'rig.sensor_count', 'comparison': '==', 'threshold': 2}]), 0),
        ('fail', f('fail.yaml', [{'metric': 'rig.camera_count', 'comparison': '<=', 'threshold': 2},
                                 {'metric': 'rig.camera_count', 'comparison': '>', 'threshold': 2}]), 1),
        ('not evaluated', f('ne.yaml', [{'metric': 'bandwidth.over_budget', 'comparison': '<=', 'threshold': 0}]), 1),
        ('unknown metric', f('unknown.yaml', [{'metric': 'no.such_metric', 'threshold': 1}]), 2),
        ('bad threshold', f('thr.yaml', [{'metric': 'rig.camera_count', 'threshold': 'two'}]), 2),
        ('wrong unit', f('unit.yaml', [{'metric': 'sync.skew_ms', 'threshold': 0.002, 'unit': 's'}]), 2),
        ('broken YAML', f('broken.yaml', text='requirements: [{metric: rig.camera_count'), 2),
        ('missing file', os.path.join(folder, 'none.yaml'), 2),
    ]


@check('requirements.gate_exit_codes', 'requirements', 'The --check gate\'s exit codes on synthetic cases',
       'cases (pass, fail, not evaluated, unknown metric, bad threshold, wrong unit, broken YAML, missing file) whose '
       'exit code differs from the one spec §11 gives',
       'the exit code of each case written out by hand from spec §11: 0 pass, 1 fail (or not evaluated), 2 error',
       0.0, 'cases', refs=('posix_exit', 'pyyaml'), cost='cheap')
def gate_exit_codes(ctx):
    import contextlib
    import io

    from ..app import main
    proj, cases = _gate_cases(ctx.tmp())
    rows, bad = [], 0
    for name, path, want in cases:
        out = os.path.join(ctx.tmp(), f'{name.replace(" ", "_")}.json')
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            got = main([proj, '--check', path, out])
        tb = 'Traceback' in err.getvalue()
        bad += int(got != want or tb)
        rows.append({'case': name, 'expected': want, 'exit': got, 'json_written': os.path.isfile(out),
                     'traceback': tb})
    return Outcome(float(bad), {'cases': rows})


# ====================================================================== bounds
@check('requirements.bound_decisions', 'requirements', 'A figure stated as a bound decides only what it settles',
       'random (figure, bound, comparison, threshold) cases where the scorecard\'s decision differs from the Monte '
       'Carlo decision over the true figures the bound allows',
       'Monte Carlo: 64 true figures per case drawn from what the bound allows (with the bound and the threshold '
       'themselves), each compared by the definition written out again; pass if all pass, fail if all fail',
       0.0, 'cases', refs=('metropolis_ulam1949',), cost='cheap')
def bound_decisions(ctx):
    from ..core import requirements as RQ
    rng = np.random.default_rng([int(ctx.settings.seed), 11])
    n = 1000 if ctx.quick else 4000
    bad, examples = 0, []
    for _ in range(n):
        op = OPS[int(rng.integers(len(OPS)))]
        bound = (None, 'at_least', 'at_most')[int(rng.integers(3))]
        t = float(np.round(rng.normal(0, 10), int(rng.integers(0, 3))))
        v = t if rng.random() < 0.15 else float(np.round(t + rng.normal(0, 5), 2))
        span = abs(t - v) + abs(t) + 10.0
        if bound is None:
            trues = np.array([v])
        elif bound == 'at_least':
            trues = np.concatenate([[v, v + span], v + rng.random(60) * span, [t, t + 1e-6 * max(1, abs(t))]])
            trues = trues[trues >= v]
        else:
            trues = np.concatenate([[v, v - span], v - rng.random(60) * span, [t, t - 1e-6 * max(1, abs(t))]])
            trues = trues[trues <= v]
        res = [_passes(float(x), op, t) for x in trues]
        want = 'pass' if all(res) else 'fail' if not any(res) else 'not_evaluated'
        got = RQ.decide(v, bound, op, t)[0]
        if got != want:
            bad += 1
            if len(examples) < 10:
                examples.append({'value': v, 'bound': bound, 'op': op, 'threshold': t, 'scorecard': got,
                                 'monte_carlo': want})
    return Outcome(float(bad), {'cases': n, 'disagreements': examples})


# ====================================================================== metrics
def _lit_ur5e():
    from ..core.robot.samples import sample_project
    from ..core.studies import bandwidth as BW
    p = sample_project('ur5e')
    BW.apply_suggestion(p, BW.budget(p)['suggestion'])
    return p


@check('requirements.metrics_vs_results', 'requirements', 'Every metric is its study\'s own figure',
       'largest difference between a metric through core/metrics.evaluate and the same figure read off the study\'s '
       'result by hand (bandwidth, time sync, cables, coverage, depth, motion blur, brackets, rig inventory on the '
       'UR5e sample with its suggested host)',
       'the figures read off each result document here, by their paths in the study\'s own output',
       0.0, 'figure units', refs=('rfc8259',), cost='study')
def metrics_vs_results(ctx):
    from ..core import jobs
    from ..core import library as LB
    from ..core import metrics as M
    p = _lit_ur5e()
    cache = jobs.ResultCache(32)
    d = {sid: jobs.run_study(sid, p, cache=cache) for sid in ('bandwidth', 'time_sync', 'cables', 'coverage',
                                                              'depth_validity', 'motion_blur', 'brackets', 'rig')}
    c0 = d['bandwidth']['controllers'][0]
    sy, cb, cv, mb = d['time_sync']['summary'], d['cables']['summary'], d['coverage']['summary'], \
        d['motion_blur']['summary']
    on = [s for s in p.variant.sensors if s.enabled]
    want = {
        'bandwidth.margin': c0['margin_mbps'], 'bandwidth.busiest_load': d['bandwidth']['headline']['busiest_load_pct'],
        'bandwidth.over_budget': float(c0['load'] > 1), 'sync.skew_ms': sy['worst_skew_s'] * 1000,
        'sync.label_error_mm': sy['label_error_m'] * 1000, 'cables.worst': cb['cables'] - cb['clear'],
        'cables.pull_snag': cb['flags'].get('pull', 0) + cb['flags'].get('snag', 0),
        'coverage.covered': cv['seen_ge1'] * 100, 'coverage.blind': cv['blind'] * 100,
        'depth.valid': d['depth_validity']['rig']['valid_pct'], 'motion_blur.worst_px': mb['smear_px'],
        'brackets.not_ready': d['brackets']['rig']['brackets'] - d['brackets']['rig']['ready']
        + d['brackets']['rig']['unset'],
        'rig.camera_count': sum(s.kind in ('camera', 'depth') for s in on),
        'rig.mass_g': sum(LB.get(s.model_id).data['mass_g'] for s in on)}
    got = M.evaluate(p, list(want), cache=cache)
    rows, worst = [], 0.0
    for mid, w in want.items():
        g = got[mid]['value']
        err = math.inf if g is None else abs(g - float(w))
        worst = max(worst, err)
        rows.append({'metric': mid, 'by_hand': w, 'metric_value': g, 'bound': got[mid].get('bound')})
    return Outcome(worst, {'metrics': rows})


# ====================================================================== reproduce
def _recorded(ctx):
    """Two study results of the UR5e sample written as batch --sample ur5e --bandwidth / --time-sync would."""
    import contextlib
    import io

    from ..app import main
    out = []
    for flag in ('--bandwidth', '--time-sync'):
        path = os.path.join(ctx.tmp(), flag.strip('-') + '.json')
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(['--sample', 'ur5e', flag, path])
        out.append((flag, path, code))
    return out


@check('requirements.reproduce_round_trip', 'requirements', '--reproduce re-runs two studies to identical numbers',
       'numbers that differ, are missing or new, and texts that changed, between two recorded results (bandwidth, '
       'time sync on the UR5e sample) and their re-runs from the recorded provenance alone',
       'the recorded result files themselves (written by batch mode with their provenance)',
       0.0, 'values', refs=('fips180_4',), cost='study')
def reproduce_round_trip(ctx):
    from ..core import reproduce as RP
    bad, rows = 0, []
    for flag, path, code in _recorded(ctx):
        rep = RP.reproduce_file(path)
        c = rep.get('counts') or {}
        n = c.get('differs', 0) + c.get('missing', 0) + c.get('extra', 0) + c.get('texts_changed', 0)
        n += int(code != 0) + int(rep['verdict'] != 'reproduces')
        bad += n
        rows.append({'result': flag, 'verdict': rep['verdict'], 'numbers': c.get('numbers'), 'exact': c.get('exact'),
                     'project': rep.get('how')})
    return Outcome(float(bad), {'results': rows})


@check('requirements.reproduce_detects_perturbation', 'requirements', '--reproduce finds a number changed by one '
       'part in a million', 'recorded results (bandwidth, time sync) with one number changed by 1e-6 relative and '
       'their result hash written again, whose reproduction does not report exactly that number as the mismatch',
       'the number changed here (its path and both values known)', 0.0, 'results', refs=('fips180_4',),
       cost='study')
def reproduce_detects_perturbation(ctx):
    from ..core import provenance as PV
    from ..core import reproduce as RP
    bad, rows = 0, []
    for flag, path, _code in _recorded(ctx):
        with open(path, encoding='utf-8') as fh:
            doc = json.load(fh)
        leaves = [(k, v) for k, v in PV._leaves({k: v for k, v in doc.items() if k != PV.KEY})
                  if isinstance(v, float) and v != 0 and 'seconds' not in k and not k.startswith('settings')
                  and not PV._UID.search(k)]
        key, val = leaves[len(leaves) // 2]
        _set(doc, key, val * (1 + 1e-6))
        doc[PV.KEY]['result_sha256'] = PV.result_digest(doc)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(doc, fh)
        rep = RP.reproduce_file(path)
        paths = [m['path'] for m in rep.get('mismatches') or []]
        ok = rep['verdict'] == 'does_not_reproduce' and paths == [key] and RP.exit_code(rep) == 1 \
            and rep['integrity']['intact'] is True
        bad += int(not ok)
        rows.append({'result': flag, 'changed': key, 'recorded': val, 'mismatches': paths[:5],
                     'verdict': rep['verdict']})
    return Outcome(float(bad), {'results': rows})


def _set(doc, path, value):
    import re
    parts = [p for p in re.split(r'\.|(\[\d+\])', path) if p]
    cur = doc
    for i, p in enumerate(parts):
        k = int(p[1:-1]) if p.startswith('[') else p
        if i == len(parts) - 1:
            cur[k] = value
        else:
            cur = cur[k]
