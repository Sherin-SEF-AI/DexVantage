"""Data-consistency checks of the library's cited prices (core/prices.py, docs/models/library.md#prices) and of the
rig cost added up from them (core/studies/rig_inventory.py, metric rig.cost). Model 'library'.

library.prices_sound — every built-in price block, read again from the entry files with the json module (not through
the library loader or core/prices): the amount a positive finite number (or null with priced options), the currency
one of the ISO 4217 codes of SIX's list one (published 2026-09-17), unit 'each', a source URL (https), the kind one of
maker_store, maker_price_list or reseller, and the date read a real date, not in the future and no more than 730 days
(two years, a default) before the day the check runs. The error is the number of price blocks that break a rule;
tolerance 0: one mistyped or stale price is one too many. core/prices.problems must find the same blocks (said in the
detail; a disagreement fails the check).

library.rig_cost_sum — the rig inventory's cost on seeded random rigs (1 to 6 sensors drawn from every built-in camera,
depth and tactile entry, priced or not) and on the built-in samples' sensor lists, against the sum written out again
here from the entry files: every part priced -> the sum, no bound; some priced -> the sum of those, bound 'at_least'
and the unpriced parts counted; none -> not stated. The error is the largest difference in the currency's units, and a
wrong bound, count or "not stated" counts as infinite; tolerance 0.005 (the total is rounded to cents).

Known limits: the checks test the data and the arithmetic, not whether a store still asks the price (prices change;
the date read says when it was read) nor whether a page really is the maker's own store (that is said, per entry, in
'kind' and 'what').
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import math
import os
import re

from .suite import Outcome, check

AGE_DAYS = 730
_KINDS = {'maker_store', 'maker_price_list', 'reseller'}
_DATE = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def _entry_files() -> dict:
    from ..core import library as L
    root = os.path.join(L.RESOURCES, 'library')
    out = {}
    for f in sorted(glob.glob(os.path.join(root, '*', '*.json'))):
        with open(f, encoding='utf-8') as fh:
            d = json.load(fh)
        out[d['id']] = d
    return out


def _iso_codes() -> set:
    from ..core import prices as PR
    return set(PR.ISO_4217)


def _rule_breaks(p: dict, codes: set, today: dt.date) -> list:
    out = []
    a = p['amount'] if 'amount' in p else p.get('value', 'missing')
    if a is None:
        if not p.get('options'):
            out.append('null amount without options')
    elif isinstance(a, bool) or not isinstance(a, (int, float)) or not math.isfinite(a) or a <= 0:
        out.append(f'amount {a!r}')
    for o in p.get('options') or []:
        oa = o.get('amount')
        if isinstance(oa, bool) or not isinstance(oa, (int, float)) or not oa > 0:
            out.append(f'option amount {oa!r}')
    if p.get('currency') not in codes:
        out.append(f'currency {p.get("currency")!r}')
    if p.get('unit') != 'each':
        out.append(f'unit {p.get("unit")!r}')
    if not str(p.get('source') or '').startswith('https://'):
        out.append('no https source')
    if p.get('kind') not in _KINDS:
        out.append(f'kind {p.get("kind")!r}')
    r = str(p.get('read') or '')
    if not _DATE.match(r):
        out.append('no date read')
    else:
        try:
            age = (today - dt.date(int(r[:4]), int(r[5:7]), int(r[8:10]))).days
        except ValueError:
            out.append(f'date {r} is not a date')
        else:
            if age < 0 or age > AGE_DAYS:
                out.append(f'read {r} ({age} days ago)')
    return out


@check('library.prices_sound', 'library', 'Cited prices: positive, ISO 4217, dated, sourced',
       'built-in price blocks that break a rule (amount > 0, ISO 4217 currency, unit each, https source, kind, date '
       'read within two years and not in the future)',
       'the rules written out again over the entry files read with the json module; currency codes from ISO 4217 '
       'list one (SIX, published 2026-09-17)', 0, 'entries', refs=('iso4217',))
def _prices_sound(ctx):
    from ..core import library as L
    from ..core import prices as PR
    today = dt.date.today()
    codes = _iso_codes()
    files = _entry_files()
    bad, priced = {}, 0
    for i, d in files.items():
        p = d.get('price')
        if p is None:
            continue
        priced += 1
        why = _rule_breaks(p, codes, today) if isinstance(p, dict) else ['not an object']
        if why:
            bad[i] = why
    theirs = {e.id for e in L.entries() if not e.is_custom and PR.block(e) is not None
              and PR.problems(PR.block(e), today)}
    agree = theirs == set(bad)
    return Outcome(float(len(bad)), {'entries': len(files), 'priced': priced, 'broken': bad,
                                     'core_prices_agrees': agree, 'max_age_days': AGE_DAYS},
                   ok=agree and priced > 0,
                   note=f'{priced} of {len(files)} built-in entries state a price; the two-year age limit is a '
                        'default.')


def _sum_again(model_ids, files) -> tuple:
    """(value, bound, n_missing, currencies) written out again from the entry files."""
    vals, miss, curs = [], 0, set()
    for m in model_ids:
        p = (files.get(m) or {}).get('price') if m else None
        a = None if not isinstance(p, dict) else p.get('amount', p.get('value'))
        if isinstance(a, (int, float)) and not isinstance(a, bool) and a > 0 and p.get('currency'):
            vals.append(float(a))
            curs.add(p['currency'])
        else:
            miss += 1
    if not model_ids or not vals or len(curs) > 1:
        return None, None, miss, curs
    return sum(vals), ('at_least' if miss else None), miss, curs


@check('library.rig_cost_sum', 'library', 'Rig cost against the stated prices added up again',
       'largest difference (USD) between the rig inventory cost and the sum written out again from the entry files, '
       'over seeded random rigs and the samples; a wrong bound, missing-part count or not-stated reads infinite',
       'the stated amounts read from the entry files with the json module and added up here', 0.005, 'USD')
def _rig_cost_sum(ctx):
    from ..core.project import Project, Sensor
    from ..core.studies import rig_inventory as RI
    files = _entry_files()
    pool = sorted(i for i, d in files.items() if d.get('kind') in ('camera', 'depth', 'tactile'))
    rigs = []
    rng = ctx.rng('library.rig_cost_sum')
    for _ in range(12 if ctx.quick else 60):
        n = int(rng.integers(1, 7))
        rigs.append(('random', [pool[int(k)] for k in rng.integers(0, len(pool), n)]))
    from ..core.robot import samples as SM
    for name, (_title, placeholder) in SM.SAMPLES.items():
        if placeholder is None:
            rigs.append((name, [s.model_id for s in SM.sample_project(name).variant.sensors if s.enabled]))
    worst, rows, priced_rigs, bound_rigs = 0.0, [], 0, 0
    for label, ids in rigs:
        p = Project('Price check')
        p.variant.sensors = [Sensor(name=f'S{k}', model_id=m) for k, m in enumerate(ids)]
        c = RI.inventory(p)['cost']
        want, bound, miss, _curs = _sum_again(ids, files)
        if want is None:
            err = 0.0 if c['value'] is None else math.inf
        else:
            err = abs(float(c['value'] or math.nan) - want) if c['value'] is not None else math.inf
            if c.get('bound') != bound or len(c['missing']) != miss:
                err = math.inf
            priced_rigs += 1
            bound_rigs += bound is not None
        if not math.isfinite(err) or math.isnan(err):
            err = math.inf
        worst = max(worst, err)
        rows.append({'rig': label, 'sensors': ids, 'cost': c['value'], 'bound': c.get('bound'), 'again': want,
                     'missing': miss})
    return Outcome(worst, {'rigs': len(rigs), 'priced': priced_rigs, 'lower_bounds': bound_rigs, 'checked': rows[:20]},
                   ok=priced_rigs > 0,
                   note=f'{len(rigs)} rigs: {priced_rigs} with a stated (or lower-bound) cost, {bound_rigs} of them '
                        'lower bounds.')
