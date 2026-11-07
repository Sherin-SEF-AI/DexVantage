"""Cited prices of the hardware library (docs/models/library.md#prices): the 'price' block of an entry, the one rule
for whether a price is stated, its checks, and the rig cost added up from the stated prices.

    stated(entry)                 -> (amount, currency) or (None, None)
    block(entry)                  -> the entry's price block (dict) or None
    problems(price, today=None)   -> [plain sentences]: what is wrong with a price block ([] when it is sound)
    total(rows)                   -> {'value', 'unit', 'bound', 'stated', 'missing', 'reason', 'currencies'}
    summary(entries=None)         -> how many entries state a price, by kind, currency and source kind
    ISO_4217                      the active ISO 4217 alphabetic codes (SIX list one, published 2026-09-17)

The price block (library.schema.json 'price'):

    {"amount": 375, "currency": "USD", "unit": "each", "kind": "maker_store",
     "source": "https://store.example/product", "read": "YYYY-MM-DD", "region": "US",
     "what": "the page's words or figure", "min_quantity": 1, "options": [{"label", "amount"}, ...]}

amount is the price of one unit (unit 'each') in currency, or null when the price depends on an option the entry does
not fix (a cable sold in several lengths: options lists each); kind says where the figure comes from:
'maker_store' (a list price from the maker's own online store), 'maker_price_list' (the maker's official price list
page, such as NVIDIA's Jetson FAQ, where min_quantity says when it is a volume price) or 'reseller' (only where the
maker sells through that store alone, said in 'what'). Nothing is estimated, and no exchange rate is applied: a total
over prices in more than one currency is refused with a sentence. A custom entry written before the block was
described may give 'value' instead of 'amount'; it reads the same.

The rig cost (core/studies/rig_inventory.py, metric rig.cost) adds up the stated prices of the rig's parts. When some
parts state no price it is a lower bound: the sum of the stated prices with bound 'at_least' and the parts that state
none named and counted (the requirements scorecard decides "cost <= X" on it only when the bound settles it; the
optimiser uses the cost objective only when every device it can choose states a price). When no part states a price
there is no bound worth giving and the cost is not stated.

Known limits: prices are list prices on the date read, without tax, shipping, duty, discounts or sales; a store can
show a different price in another region; a volume price (min_quantity > 1) is not what one unit costs; the rig cost
counts only the parts the rig inventory counts. Prices go stale: the data-consistency check fails a price read more
than two years before the day it runs.
"""
from __future__ import annotations

import datetime as _dt
import math
import re

KINDS = ('maker_store', 'maker_price_list', 'reseller')
KIND_WORDS = {'maker_store': "list price from the maker's own store",
              'maker_price_list': "the maker's official price list",
              'reseller': 'a reseller (the maker sells only through it)'}
UNIT = 'each'
MAX_AGE_DAYS = 730                 # a default: a price read more than two years ago is stale (the check fails it)
_DATE = re.compile(r'^[0-9]{4}-[0-9]{2}-[0-9]{2}$')

# ISO 4217 list one (current currency and funds code list), SIX Financial Information, the ISO 4217 maintenance
# agency, published 2026-09-17: https://www.six-group.com/dam/download/financial-information/data-center/
# iso-currrency/lists/list-one.xml (read 2026-11-07). The alphabetic codes, fund and metal codes included.
ISO_4217 = frozenset((
    'AED AFN ALL AMD AOA ARS AUD AWG AZN BAM BBD BDT BHD BIF BMD BND BOB BOV BRL BSD BTN BWP BYN BZD CAD CDF CHE CHF '
    'CHW CLF CLP CNY COP COU CRC CUP CVE CZK DJF DKK DOP DZD EGP ERN ETB EUR FJD FKP GBP GEL GHS GIP GMD GNF GTQ GYD '
    'HKD HNL HTG HUF IDR ILS INR IQD IRR ISK JMD JOD JPY KES KGS KHR KMF KPW KRW KWD KYD KZT LAK LBP LKR LRD LSL LYD '
    'MAD MDL MGA MKD MMK MNT MOP MRU MUR MVR MWK MXN MXV MYR MZN NAD NGN NIO NOK NPR NZD OMR PAB PEN PGK PHP PKR PLN '
    'PYG QAR RON RSD RUB RWF SAR SBD SCR SDG SEK SGD SHP SLE SOS SRD SSP STN SVC SYP SZL THB TJS TMT TND TOP TRY TTD '
    'TWD TZS UAH UGX USD USN UYI UYU UYW UZS VED VES VND VUV WST XAD XAF XAG XAU XBA XBB XBC XBD XCD XCG XDR XOF XPD '
    'XPF XPT XSU XTS XUA XXX YER ZAR ZMW ZWG').split())
ISO_4217_SOURCE = ('ISO 4217 list one, SIX Financial Information (maintenance agency), published 2026-09-17, '
                   'https://www.six-group.com/dam/download/financial-information/data-center/iso-currrency/lists/'
                   'list-one.xml (read 2026-11-07)')


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def block(entry) -> dict | None:
    """The price block of a library entry (an Entry or its dict), or None."""
    if entry is None:
        return None
    d = getattr(entry, 'data', entry)
    p = d.get('price') if isinstance(d, dict) else None
    return p if isinstance(p, dict) else None


def amount_of(p) -> float | None:
    """The stated amount of a price block ('amount', or the older 'value'), None when not stated or not positive."""
    if not isinstance(p, dict):
        return None
    v = _num(p.get('amount')) if 'amount' in p else _num(p.get('value'))
    return float(v) if v is not None and v > 0 else None


def stated(entry) -> tuple:
    """(amount, currency) of an entry's stated price, or (None, None): the one rule for whether a price is stated
    (the rig inventory, rig.cost and the optimiser's cost objective use it). A block without a currency reads as not
    stated: no currency is assumed."""
    p = block(entry)
    a = amount_of(p)
    cur = str(p.get('currency') or '').strip().upper() if p else ''
    if a is None or not cur:
        return None, None
    return a, cur


def problems(p, today: _dt.date | None = None, max_age_days: int = MAX_AGE_DAYS) -> list:
    """Plain sentences saying what is wrong with a built-in price block ([] when it is sound): a positive amount (or
    null with options), an ISO 4217 currency, unit 'each', a source URL, the date read (YYYY-MM-DD, not in the future,
    within max_age_days of today) and the kind."""
    if not isinstance(p, dict):
        return ['the price must be an object']
    out = []
    today = today or _dt.date.today()
    has_amount = 'amount' in p or 'value' in p
    a = p.get('amount', p.get('value'))
    opts = p.get('options') or []
    if not has_amount:
        out.append('the price states no amount')
    elif a is None:
        if not opts:
            out.append('the amount is null but no options are listed')
    elif _num(a) is None or a <= 0:
        out.append(f'the amount {a!r} is not a positive number')
    for i, o in enumerate(opts):
        oa = o.get('amount') if isinstance(o, dict) else None
        if _num(oa) is None or oa <= 0:
            out.append(f'option {i + 1} has no positive amount')
    cur = p.get('currency')
    if not isinstance(cur, str) or cur not in ISO_4217:
        out.append(f'the currency {cur!r} is not an ISO 4217 code')
    if p.get('unit') != UNIT:
        out.append(f"the unit must be '{UNIT}' (it is {p.get('unit')!r})")
    src = p.get('source')
    if not isinstance(src, str) or not src.startswith(('https://', 'http://')):
        out.append('the price has no source URL')
    if p.get('kind') not in KINDS:
        out.append(f'the kind must be one of {", ".join(KINDS)}')
    r = p.get('read')
    if not isinstance(r, str) or not _DATE.match(r):
        out.append('the date read is missing (YYYY-MM-DD)')
    else:
        try:
            d = _dt.date.fromisoformat(r)
        except ValueError:
            out.append(f'the date read {r} is not a date')
        else:
            age = (today - d).days
            if age < 0:
                out.append(f'the date read {r} is in the future')
            elif age > max_age_days:
                out.append(f'the price was read on {r}, {age} days ago: more than {max_age_days} days, so it is stale')
    mq = p.get('min_quantity')
    if mq is not None and (not isinstance(mq, int) or isinstance(mq, bool) or mq < 1):
        out.append('min_quantity must be a whole number of at least 1')
    return out


def _names(xs, n=6) -> str:
    xs = list(xs)
    return ', '.join(xs[:n]) + (f' and {len(xs) - n} more' if len(xs) > n else '')


def total(rows) -> dict:
    """The cost of the parts in rows ([{'name', 'price', 'currency'}], price None when not stated):
    {'value', 'unit', 'bound', 'stated', 'missing', 'parts', 'reason', 'currencies'} (module docstring: a lower bound
    'at_least' when some parts state no price, not stated when none does or the currencies differ)."""
    rows = list(rows)

    def has(r):                     # a price counts only with its currency (none is assumed, as in stated())
        return r.get('price') is not None and bool(str(r.get('currency') or '').strip())
    priced = [r for r in rows if has(r)]
    missing = [str(r.get('name') or '?') for r in rows if not has(r)]
    curs = sorted({str(r.get('currency')).strip().upper() for r in priced})
    out = {'value': None, 'unit': curs[0] if len(curs) == 1 else '', 'bound': None, 'stated': len(priced),
           'missing': missing, 'parts': len(rows), 'reason': '', 'currencies': curs}
    if not rows:
        out['reason'] = 'The rig has no sensor.'
        return out
    if len(curs) > 1:
        out['reason'] = (f'The prices are in different currencies ({", ".join(curs)}): no exchange rate is assumed, so '
                         'they are not added up. Use prices in one currency.')
        return out
    if not priced:
        out['reason'] = (f'A price is not stated for any of the {len(rows)} parts ({_names(missing)}), so the cost is '
                         'not stated.')
        return out
    out['value'] = round(sum(float(r['price']) for r in priced), 2)
    if missing:
        out['bound'] = 'at_least'
        out['reason'] = (f'At least {out["value"]:.2f} {out["unit"]}: a price is not stated for {len(missing)} of the '
                         f'{len(rows)} parts ({_names(missing)}), so the true cost is higher by their prices.')
    return out


def summary(entries=None) -> dict:
    """{'entries', 'priced', 'by_kind': {library kind: [priced, total]}, 'currencies', 'sources': {price kind: n}}
    over the built-in library entries (or the entries given)."""
    if entries is None:
        from . import library as LB
        entries = [e for e in LB.entries() if not e.is_custom]
    by_kind, curs, srcs, priced = {}, {}, {}, 0
    for e in entries:
        k = by_kind.setdefault(e.kind, [0, 0])
        k[1] += 1
        p = block(e)
        if p is None:
            continue
        a, cur = stated(e)
        if a is not None or p.get('options'):
            k[0] += 1
            priced += 1
            c = str(p.get('currency') or '')
            curs[c] = curs.get(c, 0) + 1
            srcs[str(p.get('kind') or '')] = srcs.get(str(p.get('kind') or ''), 0) + 1
    return {'entries': len(entries), 'priced': priced, 'by_kind': by_kind, 'currencies': curs, 'sources': srcs}
