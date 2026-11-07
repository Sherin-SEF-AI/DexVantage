"""Statistics and power analysis (research tools part 2): is a difference between two rigs, two sets of runs or two
policies' success rates real, and how many episodes or draws does it take to tell? The maths is core/stats.py, the
sources core/statcompare.py (written up in docs/models/stats.md).

    @jobs.study('stats', 'Statistics', quick=False, batch='stats-saved')   -> --stats-saved OUT.json
    dexvantage [P.dexv] --stats SPEC.json [--stats-out OUT.json]            a spec file (batch_stats.py)

Settings, project.analysis.studies['stats'] (a spec file has the same layout, schema 'dexvantage.stats/1'; every
number a labelled default unless set):

    source       'rates' (success counts, the default), 'variants' (the tolerance Monte Carlo on two variants),
                 'experiments' (a designed experiment's runs), 'tracker' (runs of the experiment tracker) or 'samples'
    rates        [{'condition', 'successes', 'trials'}, ...]; compare 'reference' (each against the first, default)
                 or 'pairs'; rate_test 'fisher' (default) or 'z'
    variants     {'a', 'b' (names or uids; default the active variant and the first other), 'metrics' (default the
                 tolerance study's: the requirements' figures), 'n' (draws, default 32), 'seed' (0), 'scale' (1),
                 'distribution' ('uniform')}
    experiment   {'mode': 'factor' (default) or 'variants', 'factor', 'levels' [a, b], 'responses',
                 'result' (an experiment result: a path or the result itself; default: the saved experiment, run),
                 'a', 'b' (mode variants)}
    tracker      {'folder' (default the project's run store), 'a', 'b' (a run id, a list of ids, or {'tag': t}),
                 'metrics'}
    samples      {'rows': [{'id', 'title', 'unit', 'better', 'a': [..], 'b': [..]}], 'paired': bool, 'names'}
                 (or 'a', 'b' and 'title' for one figure)
    alpha (0.05), confidence (0.95), correction ('holm'; 'bonferroni', 'bh', 'by', 'none'), n_boot (9999),
    n_perm (9999), seed (0), power_target (0.80)
    power        the power calculator: {'kind': 'rates' | 't2' | 'paired_t' | 'demonstrations', 'p1', 'p2', 'd',
                 'method' ('exact' default for rates, 'normal', 'arcsine', 'simulation'), 'test' ('fisher'), 'n'
                 (episodes planned per condition: its power is given), 'demo_kind' ('continuous' | 'binary'),
                 'icc', 'per_operator', 'attrition'}

The result: the comparisons (each: the test and why, the figures of both sides, the difference with its interval,
the effect size with its interval, the reference tests, the p-value and its correction across the comparisons, a
plain-English verdict with the assumptions), the power calculation with its curve, the headline (the comparison with
the smallest corrected p, as its difference B − A, with a status word: clear difference / no clear difference /
cannot test), and the suggestion (the page's Apply): when a difference is not clear, the episodes, draws or runs
that would detect the observed difference with the target power, written into the settings as the plan.

Known limits: those of core/stats.py and core/statcompare.py: fixed-sample tests only (sequential testing and early
stopping are out of scope), independence of episodes and runs, the paired tests' pairing as stated, power for rates
and normal data only, and a non-significant difference is never read as equivalence. The suggestion plans from the
observed difference, which is itself uncertain: plan for the smallest difference that matters when you know it.
"""
from __future__ import annotations

import json
import math
import os

from .. import jobs
from .. import stats as S
from .. import statcompare as SC

SID = 'stats'
SCHEMA = 'dexvantage.stats/1'          # a spec file
DEFAULTS = {'source': 'rates', 'alpha': S.ALPHA, 'confidence': S.CONFIDENCE, 'correction': S.CORRECTION,
            'n_boot': S.N_BOOT, 'n_perm': S.N_PERM, 'seed': S.SEED, 'power_target': S.POWER, 'compare': 'reference',
            'rate_test': 'fisher'}
VARIANT_DEFAULTS = {'n': 32, 'seed': 0, 'scale': 1.0, 'distribution': 'uniform'}
MAX_DRAWS = 5000
LIMITS = {'n_boot': (200, 200_000), 'n_perm': (99, 200_000), 'seed': (0, 2 ** 31 - 1)}
POWER_KINDS = ('rates', 't2', 'paired_t', 'demonstrations')


def _num(v, key, lo=None, hi=None, integer=False):
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise S.StatsError(f'The statistics setting {key} must be a number; it is {v!r}.') from None
    if not math.isfinite(x) or (lo is not None and x < lo) or (hi is not None and x > hi):
        rng = f' between {lo:g} and {hi:g}' if lo is not None and hi is not None else ''
        raise S.StatsError(f'The statistics setting {key} must be a number{rng}; it is {v!r}.')
    return int(round(x)) if integer else x


def settings_of(project_or_dict) -> dict:
    """The settings with every default filled in, and which keys took their default ('defaults')."""
    raw = project_or_dict if isinstance(project_or_dict, dict) else \
        dict(project_or_dict.analysis.studies.get(SID) or {})
    if not isinstance(raw, dict):
        raise S.StatsError('The statistics settings must be a JSON object.')
    sch = raw.get('schema')
    if sch not in (None, SCHEMA):
        raise S.StatsError(f'The statistics spec says schema "{sch}"; this version reads "{SCHEMA}".')
    st = dict(raw)
    used = []
    for k, v in DEFAULTS.items():
        if st.get(k) is None:
            st[k] = v
            used.append(k)
    if st['source'] not in SC.SOURCES:
        raise S.StatsError(f'The source must be one of {", ".join(SC.SOURCES)}; it is {st["source"]!r}.')
    st['alpha'] = _num(st['alpha'], 'alpha', 1e-6, 0.4999)
    st['confidence'] = _num(st['confidence'], 'confidence', 0.5, 0.9999)
    st['power_target'] = _num(st['power_target'], 'power_target', 0.05, 0.9999)
    for k in ('n_boot', 'n_perm', 'seed'):
        st[k] = _num(st[k], k, *LIMITS[k], integer=True)
    if st['correction'] not in S.CORRECTIONS:
        raise S.StatsError(f'The correction must be one of {", ".join(S.CORRECTIONS)}; it is {st["correction"]!r}.')
    if st['rate_test'] not in S.RATE_TESTS:
        raise S.StatsError(f'The test for success rates must be fisher or z; it is {st["rate_test"]!r}.')
    if st['compare'] not in SC.COMPARE_MODES:
        raise S.StatsError(f'compare must be reference or pairs; it is {st["compare"]!r}.')
    v = dict(VARIANT_DEFAULTS)
    given = st.get('variants') if isinstance(st.get('variants'), dict) else {}
    for k in VARIANT_DEFAULTS:
        if given.get(k) is None:
            used.append(f'variants.{k}')
        else:
            v[k] = given[k]
    v['n'] = _num(v['n'], 'variants.n', 2, MAX_DRAWS, integer=True)
    v['seed'] = _num(v['seed'], 'variants.seed', 0, 2 ** 31 - 1, integer=True)
    v['scale'] = _num(v['scale'], 'variants.scale', 0, 100)
    if v['distribution'] not in ('uniform', 'normal'):
        raise S.StatsError(f'The tolerance distribution must be uniform or normal; it is {v["distribution"]!r}.')
    v.update({k: given.get(k) for k in ('a', 'b') if given.get(k)})
    v['metrics'] = [str(m) for m in given.get('metrics') or [] if m]
    st['variants'] = v
    st['defaults'] = used
    return st


# ====================================================================== the comparisons of each source
def _load_result(src):
    if isinstance(src, dict):
        return src
    try:
        with open(str(src), encoding='utf-8') as fh:
            return json.load(fh)
    except OSError as e:
        raise S.StatsError(f'The experiment result {os.path.basename(str(src))} could not be read: '
                           f'{e.strerror or e}.') from None
    except ValueError:
        raise S.StatsError(f'The experiment result {os.path.basename(str(src))} is not valid JSON.') from None


def _samples_doc(st, data, kind, ctx=None) -> dict:
    res = SC.analyse_samples(data['rows'], bool(data.get('paired')), st['alpha'], st['confidence'],
                             st['correction'], st['n_boot'], st['n_perm'], st['seed'], (data['a'], data['b']),
                             data.get('pairing', ''), st['power_target'])
    res.update(names=[data['a'], data['b']], paired=bool(data.get('paired')), pairing=data.get('pairing', ''))
    return res


def comparisons(project, st: dict, ctx: jobs.Ctx | None = None) -> dict | None:
    """The source's comparisons (None when the source has nothing to compare yet and the power calculator is set:
    the result is the power calculation only)."""
    ctx = ctx or jobs.Ctx()
    src = st['source']
    if src == 'rates':
        rows = st.get('rates') or []
        if not rows and st.get('power'):
            return None
        res = SC.analyse_rates(rows, st['compare'], st['rate_test'], st['alpha'], st['confidence'],
                               st['correction'], st['power_target'])
        res.update(names=[r['name'] for r in res['rates']], paired=False,
                   pairing='the conditions\' episodes are independent (not paired)')
        return res
    if src == 'samples':
        sm = st.get('samples') or {}
        if not sm and st.get('power'):
            return None
        rows = sm.get('rows')
        if not rows:
            if sm.get('a') is None or sm.get('b') is None:
                raise S.StatsError('Give the samples as rows, or as a and b with their values.')
            rows = [{'id': sm.get('id', 'value'), 'title': sm.get('title', 'Value'), 'unit': sm.get('unit', ''),
                     'better': sm.get('better', 'higher'), 'a': sm['a'], 'b': sm['b']}]
        names = sm.get('names') or ['A', 'B']
        paired = bool(sm.get('paired', False))
        data = {'rows': rows, 'a': names[0], 'b': names[1], 'paired': paired,
                'pairing': sm.get('pairing') or ('paired as given (value i of A with value i of B)' if paired else
                                                 'not paired')}
        return _samples_doc(st, data, 'samples')
    if project is None:
        raise S.StatsError(f'The source {SC.SOURCE_LABEL[src].lower()} needs a project.')
    if src == 'variants':
        v = st['variants']
        data = SC.variant_samples(project, v.get('a'), v.get('b'), v['metrics'] or None, v['n'], v['seed'],
                                  v['scale'], v['distribution'], ctx)
        data['paired'] = True
        res = _samples_doc(st, data, 'variants')
        res.update(metric_source=data['metric_source'], draws=data['done'],
                   common_random_numbers=data['common_random_numbers'], cancelled=data['cancelled'])
        for r, raw in zip(res['comparisons'], data['rows']):
            r['nominal'] = [raw.get('nominal_a'), raw.get('nominal_b')]
            if raw.get('reason'):
                r['reason'] = raw['reason']
        return res
    if src == 'experiments':
        ex = st.get('experiment') or {}
        mode = ex.get('mode') or 'factor'
        if mode == 'variants':
            from .. import experiments as EX
            design = ex.get('design') or EX.settings_of(project)
            if not (design or {}).get('factors'):
                raise S.StatsError('The experiment has no factor yet: design it on Analysis ▸ Designed experiments '
                                   'first.')
            data = SC.experiment_variants(project, design, ex.get('a'), ex.get('b'), ctx)
        elif mode == 'factor':
            if not ex.get('factor'):
                raise S.StatsError('Choose the experiment\'s factor whose two levels to compare.')
            doc = _load_result(ex['result']) if ex.get('result') else None
            if doc is None:
                doc = jobs.run_study('experiments', project, ctx=ctx)
            data = SC.experiment_samples(doc, ex['factor'], ex.get('levels'), ex.get('responses'))
        else:
            raise S.StatsError(f'The experiment mode must be factor or variants; it is {mode!r}.')
        res = _samples_doc(st, data, 'experiments')
        res.update(runs=data['runs'], method=data.get('method'))
        return res
    if src == 'tracker':
        tr = st.get('tracker') or {}
        if not tr.get('a') or not tr.get('b'):
            raise S.StatsError('Choose the tracker runs of each side (run ids or a tag).')
        store = SC.tracker_store(project, tr.get('folder'))
        data = SC.tracker_samples(store, tr['a'], tr['b'], tr.get('metrics'))
        res = _samples_doc(st, data, 'tracker')
        res['runs'] = data['runs']
        return res
    raise S.StatsError(f'Unknown source {src!r}.')


# ====================================================================== the power calculator
def power_of(st: dict) -> dict | None:
    """The power calculation the settings ask for (None when they ask for none)."""
    pw = st.get('power')
    if not isinstance(pw, dict) or not pw:
        return None
    kind = pw.get('kind') or 'rates'
    if kind not in POWER_KINDS:
        raise S.StatsError(f'The power calculation must be one of {", ".join(POWER_KINDS)}; it is {kind!r}.')
    alpha = float(pw.get('alpha', st['alpha']))
    target = float(pw.get('power', st['power_target']))
    planned = pw.get('n')
    planned = None if planned in (None, '') else _num(planned, 'power.n', 2, S.N_MAX, integer=True)
    if kind == 'rates':
        if pw.get('p1') is None or pw.get('p2') is None:
            raise S.StatsError('The power calculation for success rates needs both rates (p1 and p2).')
        method = pw.get('method') or 'exact'
        if method not in S.POWER_METHODS:
            raise S.StatsError(f'The power method must be one of {", ".join(S.POWER_METHODS)}; it is {method!r}.')
        r = S.episodes_needed(pw['p1'], pw['p2'], alpha, target, method, pw.get('test') or 'fisher',
                              bool(pw.get('continuity', True)), int(pw.get('n_sim') or S.SIM_DEFAULT),
                              int(pw.get('seed') or st['seed']))
        if planned is not None:
            m = r['method'] if not (r['method'] == 'exact' and planned > S.EXACT_N_MAX) else 'normal'
            r['planned'] = {'n': planned, 'power': S.power_two_proportions(
                r['p1'], r['p2'], planned, planned, alpha, m, pw.get('test') or 'fisher',
                bool(pw.get('continuity', True)), int(pw.get('n_sim') or S.SIM_DEFAULT), st['seed'])}
        return r
    if kind in ('t2', 'paired_t'):
        if pw.get('d') is None:
            raise S.StatsError('The t-test power calculation needs the standardised effect d (difference / σ).')
        r = S.n_t(float(pw['d']), alpha, target, paired=kind == 'paired_t')
        r['curve'] = S.power_curve(kind, S.curve_ns(r['n_per_group']), alpha, d=float(pw['d']))
        r['answer'] = (f'{r["n_per_group"]} {"pairs" if kind == "paired_t" else "per group"} give '
                       f'{100 * r["power_at_n"]:.1f} % power at alpha {alpha:g} (two-sided).')
        r['out_of_scope'] = S.OUT_OF_SCOPE
        if planned is not None:
            r['planned'] = {'n': planned, 'power': S.power_t(float(pw['d']), planned, alpha, kind == 'paired_t')}
        return r
    dk = pw.get('demo_kind') or 'continuous'
    r = S.demonstrations_needed(dk, d=pw.get('d'), difference=pw.get('difference'), sd=pw.get('sd'),
                                p1=pw.get('p1'), p2=pw.get('p2'), alpha=alpha, power=target,
                                paired=bool(pw.get('paired', False)), icc=float(pw.get('icc') or 0.0),
                                per_operator=float(pw.get('per_operator') or 1.0),
                                attrition=float(pw.get('attrition') or 0.0),
                                method=pw.get('method') if pw.get('method') in ('normal', 'exact') else 'normal')
    base = r['base']
    scale = r['n_per_group'] / max(1, base['n_per_group'])
    if base['kind'] in ('t2', 'paired_t'):
        cv = S.power_curve(base['kind'], S.curve_ns(base['n_per_group']), alpha, d=base['d'])
    else:
        cv = S.power_curve('two_proportions', S.curve_ns(base['n_per_group']), alpha, p1=base['p1'], p2=base['p2'],
                           method='normal')
    r['curve'] = {'kind': cv['kind'], 'n': [int(math.ceil(n * scale)) for n in cv['n']], 'power': cv['power'],
                  'note': 'n recorded (after the clustering and the discards) against power'}
    if planned is not None:
        eff = max(2, int(planned / scale))
        r['planned'] = {'n': planned, 'power': (S.power_t(base['d'], eff, alpha, base['kind'] == 'paired_t')
                                                if base['kind'] in ('t2', 'paired_t') else
                                                S.power_two_proportions(base['p1'], base['p2'], eff, eff, alpha,
                                                                        'normal'))}
    return r


# ====================================================================== headline and suggestion
def _fmt(v, unit=''):
    if v is None:
        return '—'
    a = abs(v)
    s = f'{v:+.3g}' if a >= 100 or a == 0 else f'{v:+.3f}' if a < 1 else f'{v:+.2f}'
    if float(s) == 0:                                   # a value that rounds to zero is written 0, never +0 or -0
        s = '0'
    return f'{s} {unit}'.strip()


def headline(doc: dict) -> dict:
    cmps = doc.get('comparisons') or []
    if cmps:
        def rank(r):
            p = r.get('p_adjusted')
            return (p is None, p if p is not None else 2.0)
        best = min(cmps, key=rank)
        v = best['verdict']
        if best.get('kind') == 'rates' and best.get('difference'):
            val, unit = 100 * best['difference']['value'], 'pp'
        else:
            val, unit = (best.get('difference') or {}).get('value'), best.get('unit', '')
        corr = doc.get('correction') or {}
        test = best.get('test_label') or best.get('test', '')
        extra = (f' Test: {test}' + (f' ({best["method"]})' if best.get('method') else '') +
                 f'; p corrected with {corr.get("label", "")} over {corr.get("m", 0)} '
                 f'comparison{"" if corr.get("m") == 1 else "s"}.') if not best.get('error') else ''
        return {'value': val, 'unit': unit, 'status': v['status'], 'word': v['word'], 'comparison': best.get('id'),
                'p_adjusted': best.get('p_adjusted'), 'text': v['text'] + extra}
    pw = doc.get('power')
    if pw:
        n = pw['n_per_group']
        plan = pw.get('planned')
        if plan:
            ok = plan['power'] >= pw['power']
            return {'value': float(n), 'unit': 'per condition', 'status': 'good' if ok else 'serious',
                    'word': 'enough planned' if ok else 'underpowered',
                    'text': (f'{pw["answer"]} The plan of {plan["n"]} gives {100 * plan["power"]:.1f} % power'
                             + ('.' if ok else f': below the target {100 * pw["power"]:.0f} %.'))}
        return {'value': float(n), 'unit': 'per condition', 'status': 'warning', 'word': 'needed',
                'text': pw['answer']}
    return {'value': None, 'unit': '', 'status': 'critical', 'word': 'nothing to compare',
            'text': 'Nothing to compare: enter success counts, pick two variants, an experiment\'s factor or two '
                    'sets of tracker runs.'}


def suggestion(doc: dict, st: dict) -> dict | None:
    """For the headline comparison when its difference is not clear: the episodes / draws / runs that detect the
    observed difference with the target power, as settings to write (the page's Apply)."""
    cid = (doc.get('headline') or {}).get('comparison')
    best = next((c for c in doc.get('comparisons') or [] if c.get('id') == cid), None)
    if best is None or best.get('error') or (best.get('verdict') or {}).get('word') != 'no clear difference':
        return None
    target = st['power_target']
    try:
        if best.get('kind') == 'rates':
            p1, p2 = best['a']['rate'], best['b']['rate']
            if p1 == p2:
                return None
            n = S.n_two_proportions(p1, p2, st['alpha'], target, 'normal')['n_per_group']
            if n > S.N_MAX:
                return None
            pw = {'kind': 'rates', 'p1': round(p1, 4), 'p2': round(p2, 4), 'method': 'exact' if n <= S.EXACT_N_MAX
                  else 'normal', 'n': n}
            return {'kind': 'power', 'settings': {'power': pw}, 'n': n,
                    'text': (f'To tell {100 * p1:.0f} % from {100 * p2:.0f} % apart {100 * target:.0f} % of the time, '
                             f'plan about {n} episodes per condition (the observed rates, themselves uncertain). Apply '
                             'writes this plan into the power calculator.')}
        eff = best.get('standardised') or {}
        d = eff.get('d')
        if d is None or d == 0:
            return None
        paired = bool(best.get('paired'))
        n = S.n_t(abs(d), st['alpha'], target, paired)['n_per_group']
        if st['source'] == 'variants':
            n = min(n, MAX_DRAWS)
            return {'kind': 'draws', 'settings': {'variants': {'n': n}}, 'n': n,
                    'text': (f'{best["title"]}: about {n} draws per variant would detect the observed difference '
                             f'({"d_z" if paired else "d"} = {abs(d):.2f}) {100 * target:.0f} % of the time. Apply '
                             'sets the draws; run again to test.')}
        return {'kind': 'power', 'settings': {'power': {'kind': 'paired_t' if paired else 't2', 'd': round(abs(d), 4),
                                                        'n': n}}, 'n': n,
                'text': (f'{best["title"]}: about {n} {"pairs" if paired else "runs per side"} would detect the '
                         f'observed difference ({"d_z" if paired else "d"} = {abs(d):.2f}) {100 * target:.0f} % of '
                         'the time. Apply writes this plan into the power calculator.')}
    except S.StatsError:
        return None


# ====================================================================== the study
def analyse(project, st: dict, ctx: jobs.Ctx | None = None) -> dict:
    ctx = ctx or jobs.Ctx()
    ctx.report(0.0, 'Comparing')
    res = comparisons(project, st, ctx)
    pw = power_of(st)
    if res is None and pw is None:
        raise S.StatsError('Nothing to compare: enter success counts (at least two conditions), pick two variants, an '
                           'experiment\'s factor or tracker runs, or fill in the power calculator.')
    doc = {'source': st['source'], 'source_label': SC.SOURCE_LABEL[st['source']]}
    if res is not None:
        doc.update(res)
    doc['comparisons'] = doc.get('comparisons') or []
    doc['power'] = pw
    doc['headline'] = headline(doc)
    sg = suggestion(doc, st)
    doc['suggestion'] = sg
    assumptions = []
    for c in doc['comparisons']:
        for a in c.get('assumptions') or []:
            if a not in assumptions:
                assumptions.append(a)
    if doc.get('pairing') and not any(doc['pairing'] in a for a in assumptions):
        assumptions.insert(0, doc['pairing'])
    doc['assumptions'] = assumptions
    notes = []
    for c in doc['comparisons']:
        notes += [f'{c.get("title", "")}: {n}' for n in c.get('notes') or []]
        if c.get('dropped'):
            notes.append(f'{c.get("title", "")}: {c["dropped"]} value{"" if c["dropped"] == 1 else "s"} without a '
                         'figure left out' + (' (their pairs too)' if c.get('paired') else '') +
                         (f' ({c["reason"]})' if c.get('reason') else '') + '.')
    if doc.get('cancelled'):
        notes.insert(0, 'The Monte Carlo was stopped early: the comparisons cover the draws made.')
    doc['notes'] = notes
    doc['settings'] = {k: v for k, v in st.items() if k not in ('defaults',)}
    ex = doc['settings'].get('experiment')
    if isinstance(ex, dict) and isinstance(ex.get('result'), dict):
        doc['settings']['experiment'] = {**ex, 'result': '(the experiment result in memory)'}
    doc['defaults'] = st['defaults']
    doc['advice'] = sg['text'] if sg else ''
    doc['out_of_scope'] = S.OUT_OF_SCOPE
    doc['known_limits'] = ('Fixed-sample tests only; episodes and runs independent; pairing as stated; a difference '
                           'that is not clear is not shown to be absent (see the smallest detectable difference).')
    doc['references'] = list(S.REFERENCES)
    ctx.report(1.0, doc['headline']['text'])
    return doc


@jobs.study(SID, 'Statistics', quick=False, batch='stats-saved')
def run(project, ctx: jobs.Ctx) -> dict:
    return analyse(project, settings_of(project), ctx)


# ====================================================================== batch: a spec file
def read_spec(src) -> dict:
    """A spec (path, JSON text or dict) as a settings dict."""
    if isinstance(src, dict):
        return dict(src)
    text = str(src)
    if not text.lstrip().startswith('{'):
        try:
            with open(text, encoding='utf-8') as fh:
                text = fh.read()
        except OSError as e:
            raise S.StatsError(f'The statistics spec {os.path.basename(text)} could not be read: '
                               f'{e.strerror or e}.') from None
    try:
        d = json.loads(text)
    except ValueError as e:
        raise S.StatsError(f'The statistics spec is not valid JSON ({e}).') from None
    if not isinstance(d, dict):
        raise S.StatsError('The statistics spec must be a JSON object.')
    return d


def run_spec(spec, project=None, ctx: jobs.Ctx | None = None) -> tuple[dict, object]:
    """(result doc, the project it ran on) for a spec: laid over the project's saved settings on a snapshot (or a
    blank project when none is given), through core.jobs.run_study so the result carries the study's schema."""
    from ..project import blank_project
    d = read_spec(spec)
    snap = project.snapshot() if project is not None else blank_project('No project')
    if d.get('schema') not in (None, SCHEMA):
        raise S.StatsError(f'The statistics spec says schema "{d["schema"]}"; this version reads "{SCHEMA}".')
    d.pop('schema', None)
    snap.analysis.studies[SID] = d
    settings_of(snap)                                    # refuse a bad spec before anything runs
    doc = jobs.run_study(SID, snap, ctx=ctx)
    return doc, snap


def default_output(spec_path: str) -> str:
    base = spec_path[:-5] if spec_path.lower().endswith('.json') else spec_path
    return base + '.results.json'
