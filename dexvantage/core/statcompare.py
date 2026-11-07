"""Comparisons from the app's sources (research tools part 2): what core/stats.py compares, taken from

* **success counts** the user enters (a small table: condition, successes, trials): each condition against the first
  (the reference) or every pair, Fisher's exact test by default;
* **two variants over the tolerance Monte Carlo** (core/tolerance.monte_carlo run on each variant with the same seed,
  so draw i of both uses the same random numbers: the draws are paired by seed, a paired comparison per figure);
* **experiment runs**: a designed experiment's runs split by one factor's two levels (paired by the other factors'
  levels when every run has exactly one partner, as in a full factorial; else unpaired), or the saved design run on
  two variants (paired by run: the same factor values and seed);
* **tracker runs** (core/tracking): two runs' per-draw samples (a tolerance Monte Carlo logged as a run; paired when
  both state the same seed and number of draws), or two groups of runs (by run ids or a tag), one value per run,
  paired by seed when every run states a seed and the seeds match one to one;
* **samples** given as numbers (batch mode, the API).

    analyse_samples(rows, paired, ...)  -> per figure core/stats.compare_samples, the correction across figures and
                                           a verdict per figure
    analyse_rates(rows, ...)            -> each comparison of success rates, corrected across comparisons
    variant_samples(project, a, b, ...) -> the tolerance Monte Carlo's per-draw values of both variants
    experiment_samples(doc, factor, ...) / experiment_variants(project, design, a, b, ...)
    tracker_samples(store, a, b, ...)   -> from run records

Known limits: a variant comparison draws each variant's mounts from its own tolerances; with the same seed and the
same sensors in the same order the random numbers are shared (common random numbers), else draw i is paired by index
only (still a valid paired test: the draws are independent between pairs, only less sharp). A figure that has no
value in a draw (its study refused on the moved mount) drops that pair (counted). Experiment and tracker values are
the studies' own figures with their limits; one value per run gives one replicate per run, so a group needs at least
two runs. Each comparison's p-value is corrected across the figures (or comparisons) of one analysis only.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np

from . import stats as S
from .jobs import Ctx

SOURCES = ('rates', 'variants', 'experiments', 'tracker', 'samples')
SOURCE_LABEL = {'rates': 'Success counts', 'variants': 'Variants over the tolerance Monte Carlo',
                'experiments': 'Experiment runs', 'tracker': 'Tracker runs', 'samples': 'Samples given as numbers'}
COMPARE_MODES = ('reference', 'pairs')
_MC_CACHE: OrderedDict = OrderedDict()        # (variant content key, ids, n, seed, scale, dist) -> raw values
_MC_SIZE = 8


# =========================================================================== analysing samples of several figures
def analyse_samples(rows: list, paired: bool, alpha: float = S.ALPHA, confidence: float = S.CONFIDENCE,
                    correction: str = S.CORRECTION, n_boot: int = S.N_BOOT, n_perm: int = S.N_PERM,
                    seed: int = S.SEED, names=('A', 'B'), pairing: str = '', power: float = S.POWER) -> dict:
    """rows: [{'id', 'title', 'unit', 'better', 'a': [values], 'b': [values], 'dropped'?: int, 'paired'?: bool}]
    (None values drop their pair when paired, the value when not). Each row is compared (core/stats.compare_samples),
    the p-values corrected across the rows, and each gets a verdict."""
    out, pvals = [], []
    for r in rows:
        pr = bool(r.get('paired', paired))
        a, b = list(r.get('a') or []), list(r.get('b') or [])
        dropped = int(r.get('dropped') or 0)
        if pr:
            keep = [(x, y) for x, y in zip(a, b) if _ok(x) and _ok(y)]
            dropped += max(len(a), len(b)) - len(keep)
            a, b = [x for x, _ in keep], [y for _, y in keep]
        else:
            na, nb = len(a), len(b)
            a, b = [x for x in a if _ok(x)], [y for y in b if _ok(y)]
            dropped += na - len(a) + nb - len(b)
        row = {'id': r.get('id', ''), 'title': r.get('title') or r.get('id', ''), 'unit': r.get('unit', ''),
               'better': r.get('better', 'higher'), 'dropped': dropped}
        if len(a) < 2 or len(b) < 2:
            why = f' ({str(r["reason"]).rstrip(".")})' if r.get('reason') else ''
            row.update(kind='samples', paired=pr, p=None, names=list(names),
                       error=(f'only {len(a)} {"pair has" if len(a) == 1 else "pairs have"} a value on both sides'
                              if pr else f'only {len(a)} and {len(b)} values') + f'; at least 2 are needed{why}.')
            out.append(row)
            pvals.append(None)
            continue
        try:
            cmp = S.compare_samples(a, b, paired=pr, alpha=alpha, confidence=confidence, n_boot=n_boot,
                                    n_perm=n_perm, seed=seed, better=row['better'], names=names,
                                    pairing=r.get('pairing', pairing) if pr else '', power=power)
            row.update(cmp)
        except S.StatsError as e:
            row.update(kind='samples', paired=pr, p=None, error=str(e), names=list(names))
        out.append(row)
        pvals.append(row.get('p'))
    return _finish(out, pvals, alpha, correction)


def _ok(v) -> bool:
    return v is not None and not isinstance(v, bool) and math.isfinite(float(v))


def _finish(rows, pvals, alpha, correction) -> dict:
    adj = S.adjust(pvals, correction, alpha)
    for r, pa, rej in zip(rows, adj['adjusted'], adj['reject']):
        r['p_adjusted'] = pa
        r['reject'] = rej
        if r.get('error'):
            r['verdict'] = {'status': 'critical', 'word': 'cannot test', 'text': f'{r["title"]}: {r["error"]}'}
        else:
            r['verdict'] = S.verdict(r, pa, r['title'], r.get('unit', ''), alpha)
    return {'comparisons': rows, 'correction': {k: v for k, v in adj.items() if k not in ('adjusted', 'reject')}}


# =========================================================================== success counts
def clean_rates(rows) -> list:
    """[{'condition', 'successes', 'trials'}] checked (plain sentences for what cannot be read)."""
    if not isinstance(rows, (list, tuple)) or len(rows) < 2:
        raise S.StatsError('Enter at least two conditions, each with its successes and trials.')
    out, names = [], set()
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise S.StatsError(f'Row {i + 1} must give a condition, its successes and its trials.')
        name = str(r.get('condition') or r.get('name') or f'Condition {i + 1}').strip()[:80]
        if name in names:
            raise S.StatsError(f'Two rows are called "{name}": give each condition its own name.')
        names.add(name)
        k, n = S._counts(r.get('successes'), r.get('trials'), f'"{name}"')        # noqa: SLF001
        out.append({'condition': name, 'successes': k, 'trials': n})
    return out


def analyse_rates(rows, compare: str = 'reference', test: str = 'fisher', alpha: float = S.ALPHA,
                  confidence: float = S.CONFIDENCE, correction: str = S.CORRECTION, power: float = S.POWER) -> dict:
    """Every comparison of the conditions' success rates: each against the first ('reference') or every pair
    ('pairs'), corrected across the comparisons."""
    rows = clean_rates(rows)
    if compare not in COMPARE_MODES:
        raise S.StatsError(f'Compare must be reference or pairs; it is {compare!r}.')
    pairs = [(0, j) for j in range(1, len(rows))] if compare == 'reference' else \
        [(i, j) for i in range(len(rows)) for j in range(i + 1, len(rows))]
    out, pvals = [], []
    for i, j in pairs:
        a, b = rows[i], rows[j]
        c = S.compare_rates(a['successes'], a['trials'], b['successes'], b['trials'], test, confidence,
                            (a['condition'], b['condition']))
        c.update(id=f'{a["condition"]} vs {b["condition"]}', title=f'{b["condition"]} against {a["condition"]}',
                 unit='%', better='higher', alpha=alpha)
        n = min(a['trials'], b['trials'])
        c['detectable'] = {**(S.detectable_difference('two_proportions', n, alpha, power, p1=a['successes'] /
                                                      a['trials']) or {}), 'power': power}
        out.append(c)
        pvals.append(c['p'])
    res = _finish(out, pvals, alpha, correction)
    res['rates'] = [S.proportion(r['successes'], r['trials'], confidence, r['condition']) for r in rows]
    res['compare'] = compare
    return res


# =========================================================================== variants over the tolerance Monte Carlo
def find_variant(project, ident):
    """The variant named or uid'd ident (StatsError when there is none)."""
    if ident in (None, ''):
        return None
    for v in project.variants:
        if v.uid == ident or v.name == ident:
            return v
    raise S.StatsError(f'There is no variant called {ident}: the variants are '
                       f'{", ".join(v.name for v in project.variants)}.')


def pick_variants(project, a=None, b=None) -> tuple:
    """(variant A, variant B): the ones named, else A the active variant and B the first other one."""
    if len(project.variants) < 2:
        raise S.StatsError('The project has one variant: add a second one (Variants ▸ Add variant) to compare two '
                           'rigs.')
    va = find_variant(project, a) or project.variant
    vb = find_variant(project, b) or next(v for v in project.variants if v.uid != va.uid)
    if va.uid == vb.uid:
        raise S.StatsError(f'Both sides are the variant {va.name}: pick two different variants.')
    return va, vb


def _with_variant(project, v):
    p = project.snapshot()
    p.active_uid = v.uid
    p.analysis.studies.pop('stats', None)          # the Monte Carlo does not read them: a changed alpha or
    return p                                       # correction re-uses the draws (_MC_CACHE)


def _mc_key(p, ids, n, seed, scale, dist):
    from .jobs import ResultCache
    return (ResultCache().key('tolerance', p)[1:], p.active_uid, tuple(ids), int(n), int(seed), float(scale), dist)


def variant_samples(project, a=None, b=None, metric_ids=None, n: int = 32, seed: int = 0, scale: float = 1.0,
                    distribution: str = 'uniform', ctx: Ctx | None = None, evaluator=None) -> dict:
    """The tolerance Monte Carlo of variants a and b with one seed: {'a', 'b' (names), 'rows' (one per figure:
    'a', 'b' the per-draw values), 'pairing', 'common', 'metric_source'}. metric_ids default: the tolerance study's
    choice (the requirements' figures, else its defaults)."""
    from . import tolerance as TL
    from .studies import tolerance as TS
    ctx = ctx or Ctx()
    va, vb = pick_variants(project, a, b)
    pa, pb = _with_variant(project, va), _with_variant(project, vb)
    if metric_ids:
        from . import metrics as M
        M.load_metrics()
        bad = [m for m in metric_ids if m not in M.METRICS]
        if bad:
            raise S.StatsError(f'There is no metric called {", ".join(bad)}.')
        ids, why = list(dict.fromkeys(metric_ids)), 'the statistics settings'
    else:
        ids, why = TS.metric_ids(pa, {'metrics': []})
    docs = []
    for k, p in enumerate((pa, pb)):
        key = _mc_key(p, ids, n, seed, scale, distribution) if evaluator is None else None
        if key is not None and key in _MC_CACHE:
            _MC_CACHE.move_to_end(key)
            docs.append(_MC_CACHE[key])
            continue
        inner = Ctx(progress=(lambda f, t, k=k: ctx.report(0.5 * k + 0.5 * f, f'{(va, vb)[k].name}: {t}')),
                    cancel=ctx.cancel)
        d = TL.monte_carlo(p, ids, n, seed, scale, distribution, ctx=inner, evaluator=evaluator)
        slim = {'done': d['done'], 'n': d['n'], 'cancelled': d['cancelled'],
                'sources': [(s['sensor_uid'], s['source']) for s in d['sources']],
                'metrics': {m: {kk: r.get(kk) for kk in ('title', 'unit', 'better', 'values', 'nominal', 'reason')}
                            for m, r in d['metrics'].items()}}
        if key is not None and not d['cancelled']:
            _MC_CACHE[key] = slim
            while len(_MC_CACHE) > _MC_SIZE:
                _MC_CACHE.popitem(last=False)
        docs.append(slim)
    da, db = docs
    common = [k for _u, k in da['sources']] == [k for _u, k in db['sources']]
    rows = []
    for m in ids:
        ra, rb = da['metrics'].get(m) or {}, db['metrics'].get(m) or {}
        rows.append({'id': m, 'title': ra.get('title') or m, 'unit': ra.get('unit', ''),
                     'better': ra.get('better', 'higher'), 'a': ra.get('values') or [], 'b': rb.get('values') or [],
                     'nominal_a': ra.get('nominal'), 'nominal_b': rb.get('nominal'),
                     'reason': ra.get('reason') or rb.get('reason')})
    pairing = (f'the draws are paired by seed: draw i of both variants uses the same random numbers (seed {seed}'
               + (', the same sensors drawn in the same order' if common else
                  '; the variants\' sensors differ, so draw i is paired by its index only') + ')')
    return {'a': va.name, 'b': vb.name, 'a_uid': va.uid, 'b_uid': vb.uid, 'rows': rows, 'pairing': pairing,
            'common_random_numbers': common, 'metric_source': why, 'n': int(n), 'seed': int(seed),
            'done': [da['done'], db['done']], 'cancelled': da['cancelled'] or db['cancelled']}


# =========================================================================== experiment runs
def _same(u, v) -> bool:
    if isinstance(u, (int, float)) and isinstance(v, (int, float)) and not isinstance(u, bool):
        return abs(float(u) - float(v)) <= 1e-12 * max(1.0, abs(float(u)))
    return u == v


def experiment_samples(doc: dict, factor: str, levels=None, responses=None) -> dict:
    """A designed experiment's runs split by one factor: level A against level B (the levels given; else the first
    and last level of a categorical or bool factor; a continuous factor's lower half against its upper half, split
    at its median). Paired by the other factors' values when every A run has exactly one B run with the same other
    values (a full factorial), else unpaired."""
    if not isinstance(doc, dict) or not doc.get('runs'):
        raise S.StatsError('There is no experiment result to compare: run the designed experiment first (Analysis ▸ '
                           'Designed experiments).')
    finfo = {f.get('path'): f for f in doc.get('factors') or []}
    if factor not in finfo:
        raise S.StatsError(f'The experiment has no factor {factor}; its factors are {", ".join(finfo) or "none"}.')
    f = finfo[factor]
    runs = [r for r in doc['runs'] if r.get('done')]
    if (doc.get('method') == 'saltelli'):
        runs = [r for r in runs if r.get('block') in ('A', 'B')]
    vals = [r['values'].get(factor) for r in runs]
    split = ''
    if levels:
        if len(levels) != 2:
            raise S.StatsError('Give two levels of the factor to compare.')
        la, lb = levels
        ia = [i for i, v in enumerate(vals) if _same(v, la)]
        ib = [i for i, v in enumerate(vals) if _same(v, lb)]
        names = (f'{f.get("title", factor)} = {la}', f'{f.get("title", factor)} = {lb}')
    elif f.get('kind') in ('categorical', 'bool'):
        lv = f.get('levels') if f.get('kind') == 'categorical' else [False, True]
        la, lb = lv[0], lv[-1]
        ia = [i for i, v in enumerate(vals) if _same(v, la)]
        ib = [i for i, v in enumerate(vals) if _same(v, lb)]
        names = (f'{f.get("title", factor)} = {la}', f'{f.get("title", factor)} = {lb}')
    else:
        x = np.array([float(v) for v in vals])
        uniq = np.unique(x)
        if len(uniq) == 2:
            ia = [i for i, v in enumerate(x) if v == uniq[0]]
            ib = [i for i, v in enumerate(x) if v == uniq[1]]
            names = (f'{f.get("title", factor)} = {uniq[0]:g}', f'{f.get("title", factor)} = {uniq[1]:g}')
        else:
            med = float(np.median(x))
            ia = [i for i, v in enumerate(x) if v <= med]
            ib = [i for i, v in enumerate(x) if v > med]
            names = (f'{f.get("title", factor)} ≤ {med:.4g}', f'{f.get("title", factor)} > {med:.4g}')
            split = f'split at the median {med:.4g} {f.get("unit", "")}'.strip()
    if len(ia) < 2 or len(ib) < 2:
        raise S.StatsError(f'The factor {f.get("title", factor)} has {len(ia)} and {len(ib)} runs at the two levels: '
                           'each needs at least 2.')
    others = [p for p in finfo if p != factor]

    def key(i):
        return tuple(repr(runs[i]['values'].get(p)) for p in others)
    paired = False
    if others and not split:
        ka, kb = [key(i) for i in ia], [key(i) for i in ib]
        if len(set(ka)) == len(ka) and sorted(ka) == sorted(kb):
            order = {k: i for k, i in zip(kb, ib)}
            ib = [order[k] for k in ka]
            paired = True
    ids = [r.get('id') for r in doc.get('responses') or []]
    if responses:
        ids = [m for m in ids if m in responses]
    meta = {r.get('id'): r for r in doc.get('responses') or []}
    rows = []
    for m in ids:
        rows.append({'id': m, 'title': meta[m].get('title', m), 'unit': meta[m].get('unit', ''),
                     'better': meta[m].get('better', 'higher'),
                     'a': [runs[i]['responses'].get(m) for i in ia], 'b': [runs[i]['responses'].get(m) for i in ib]})
    pairing = ('the runs are paired by the other factors\' values (each run at one level has one run at the other with '
               'the same other factors)' if paired else
               'the runs are not paired' + (f' ({split})' if split else ' (no one-to-one match of the other factors)'))
    return {'a': names[0], 'b': names[1], 'rows': rows, 'paired': paired, 'pairing': pairing, 'factor': factor,
            'runs': [len(ia), len(ib)], 'method': doc.get('method_label') or doc.get('method')}


def experiment_variants(project, design, a=None, b=None, ctx: Ctx | None = None) -> dict:
    """The design run on variants a and b (core/experiments.run with design.variant set, the same seed): runs paired
    by run number (the same factor values)."""
    from . import experiments as EX
    ctx = ctx or Ctx()
    va, vb = pick_variants(project, a, b)
    docs = []
    for k, v in enumerate((va, vb)):
        d = EX.load_design(design, project)
        d.variant = v.uid
        inner = Ctx(progress=(lambda f, t, k=k, v=v: ctx.report(0.5 * k + 0.5 * f, f'{v.name}: {t}')),
                    cancel=ctx.cancel)
        docs.append(EX.run(project, d, inner))
    da, db = docs
    ids = [r['id'] for r in da['responses']]
    meta = {r['id']: r for r in da['responses']}
    rows = []
    for m in ids:
        rows.append({'id': m, 'title': meta[m]['title'], 'unit': meta[m]['unit'], 'better': meta[m]['better'],
                     'a': [r['responses'].get(m) if r['done'] else None for r in da['runs']],
                     'b': [r['responses'].get(m) if r['done'] else None for r in db['runs']]})
    return {'a': va.name, 'b': vb.name, 'rows': rows, 'paired': True,
            'pairing': f'the runs are paired by run: run i of both variants has the same factor values '
                       f'({da["method_label"]}, seed {da["design"]["design"]["seed"]})',
            'runs': [len(da['runs']), len(db['runs'])], 'method': da.get('method_label')}


# =========================================================================== tracker runs
def _run_seed(rec):
    seeds = rec.get('seeds') or {}
    if not isinstance(seeds, dict) or not seeds:
        return None
    s = seeds.get(rec.get('study')) if rec.get('study') in seeds else next(iter(seeds.values()))
    if isinstance(s, dict):
        s = s.get('seed')
    return s if isinstance(s, (int, float)) and not isinstance(s, bool) else None


def _select(store, sel) -> list:
    """Run records chosen by sel: a run id, a list of run ids, or {'tag': t} / {'study': s} (every finished run)."""
    if isinstance(sel, str):
        sel = [sel]
    if isinstance(sel, dict):
        rows = store.runs(study=sel.get('study'), tag=sel.get('tag'), status='finished')
        ids = [r['run_id'] for r in rows]
    elif isinstance(sel, (list, tuple)):
        ids = [str(x) for x in sel]
    else:
        raise S.StatsError('Choose the runs of each side by their ids or by a tag.')
    out = []
    for rid in ids:
        try:
            out.append(store.get(rid))
        except Exception as e:                       # noqa: BLE001 — the store's sentence
            raise S.StatsError(str(e)) from None
    return out


def tracker_samples(store, a, b, metrics=None) -> dict:
    """From the tracker's run store: two runs that kept per-draw samples (a tolerance Monte Carlo) compare their
    draws, paired when both state the same seed and number of draws; two groups of runs compare one value per run,
    paired by seed when every run states one and the seeds match one to one."""
    ra, rb = _select(store, a), _select(store, b)
    if not ra or not rb:
        raise S.StatsError('Each side needs at least one run of the tracker.')
    both = {r.get('run_id') for r in ra} & {r.get('run_id') for r in rb}
    if both:
        raise S.StatsError(f'Run {sorted(both)[0][:8]} is on both sides: a run compared with itself has no '
                           'difference to test; choose different runs for A and B.')

    def label(rs):
        return rs[0].get('name') or rs[0].get('run_id', '')[:8] if len(rs) == 1 else f'{len(rs)} runs'
    rows = []
    if len(ra) == 1 and len(rb) == 1 and (ra[0].get('samples') or rb[0].get('samples')):
        x, y = ra[0], rb[0]
        keys = [k for k in (x.get('samples') or {}) if k in (y.get('samples') or {})]
        if metrics:
            keys = [k for k in keys if k in metrics]
        sx, sy = _run_seed(x), _run_seed(y)
        same = sx is not None and sx == sy
        for k in keys:
            m = (x.get('metrics') or {}).get(k) or {}
            va, vb = x['samples'][k], y['samples'][k]
            rows.append({'id': k, 'title': m.get('title', k), 'unit': m.get('unit', ''),
                         'better': m.get('better') or 'higher', 'a': va, 'b': vb,
                         'paired': bool(same and len(va) == len(vb))})
        if not rows:
            raise S.StatsError('The two runs share no figure with per-draw values.')
        paired = all(r['paired'] for r in rows)
        pairing = (f'the draws are paired by seed (both runs used seed {sx})' if paired else
                   'the draws are not paired (the runs state different seeds or numbers of draws)')
        return {'a': label(ra), 'b': label(rb), 'rows': rows, 'paired': paired, 'pairing': pairing,
                'runs': [1, 1]}
    keys = []
    for r in ra + rb:
        keys += [k for k, m in (r.get('metrics') or {}).items() if k not in keys and '@' not in k and
                 isinstance(m, dict)]
    if metrics:
        keys = [k for k in keys if k in metrics]
    seeds_a, seeds_b = [_run_seed(r) for r in ra], [_run_seed(r) for r in rb]
    paired = (None not in seeds_a + seeds_b and len(set(seeds_a)) == len(seeds_a) and sorted(seeds_a) == sorted(
        seeds_b))
    if paired:
        order = {s: r for s, r in zip(seeds_b, rb)}
        rb = [order[s] for s in seeds_a]
    for k in keys:
        def val(r):
            return ((r.get('metrics') or {}).get(k) or {}).get('value')
        m = next(((r.get('metrics') or {}).get(k) for r in ra + rb if (r.get('metrics') or {}).get(k)), {})
        rows.append({'id': k, 'title': m.get('title', k), 'unit': m.get('unit', ''), 'better': m.get('better') or
                     'higher', 'a': [val(r) for r in ra], 'b': [val(r) for r in rb]})
    if not rows:
        raise S.StatsError('The runs state no figure to compare.')
    pairing = ('the runs are paired by seed (each seed run once on each side)' if paired else
               'the runs are not paired (one value per run; the seeds do not match one to one)')
    return {'a': label(ra), 'b': label(rb), 'rows': rows, 'paired': paired, 'pairing': pairing,
            'runs': [len(ra), len(rb)]}


def tracker_store(project, folder: str | None = None):
    """The project's run store (core/tracking), or StatsError with why there is none."""
    try:
        from . import tracking as TR
    except ImportError:
        raise S.StatsError('This build has no experiment tracker (core/tracking).') from None
    try:
        st = TR.store_for(project, folder or None)
    except Exception as e:                             # noqa: BLE001 — TrackingError: a plain sentence
        raise S.StatsError(str(e)) from None
    if not st.exists():
        raise S.StatsError(f'The run store {st.name} has no runs yet: turn tracking on (Runs page) and run studies.')
    return st
