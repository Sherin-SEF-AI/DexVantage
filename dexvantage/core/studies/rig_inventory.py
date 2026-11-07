"""Rig inventory (spec §7.11, §11): what the active variant's rig is made of, as the figures the optimiser and the
requirements scorecard compare: the camera count, the sensor count, and the cost, mass and power of the enabled
sensors where the library states them.

    @jobs.study('rig', 'Rig inventory', quick=False, batch='rig-inventory')   → --rig-inventory OUT.json

Each mass and power total is {'value', 'unit', 'stated', 'missing', 'reason'}: the sum over the enabled sensors when
every one states the figure, else value None ("not stated") with the devices that do not state it. The cost reads the
cited "price" block of each sensor's library entry (core/prices.py: amount, ISO 4217 currency, unit 'each', source URL,
date read, kind) and is core/prices.total: the sum when every sensor states a price; a lower bound (bound 'at_least')
with the parts that state none named and counted when only some do; not stated when none does; refused with a
sentence when the prices are in different currencies (no exchange rate is assumed).

Known limits: only the sensors are counted (hosts, hubs, cables, lights and brackets are not in the totals); a sensor
with no library device (the default camera) states nothing; prices are list prices on the date read (no tax,
shipping or discount) and many entries state none (docs/models/library.md#prices); mass and power are the maker's
figures for the device alone (no bracket, no cable).
"""
from __future__ import annotations

from .. import jobs
from .. import library as LB
from .. import prices as PR

STUDY_ID = 'rig'
CAMERA_KINDS = ('camera', 'depth')


def stated_price(entry) -> tuple:
    """(amount, currency) of a library entry's stated price, or (None, None): the one rule for whether a price is
    stated (the rig inventory and the optimiser's cost objective use it; core/prices.stated)."""
    return PR.stated(entry)


def inventory(project) -> dict:
    v = project.variant
    on = [s for s in v.sensors if s.enabled]
    rows = []
    for s in on:
        e = LB.get(s.model_id) if s.model_id else None
        price, cur = stated_price(e)
        pb = PR.block(e) if price is not None else None
        rows.append({'uid': s.uid, 'name': s.name, 'kind': s.kind, 'model_id': s.model_id or '',
                     'device': e.title if e is not None else None,
                     'price': price, 'currency': cur,
                     'price_source': (pb or {}).get('source'), 'price_read': (pb or {}).get('read'),
                     'price_kind': (pb or {}).get('kind'),
                     'mass_g': e.data.get('mass_g') if e is not None else None,
                     'power_w': e.data.get('power_w') if e is not None else None})

    def total(key, unit, what):
        missing = [r['name'] for r in rows if r[key] is None]
        if not rows:
            return {'value': None, 'unit': unit, 'stated': 0, 'missing': [], 'reason': 'The rig has no sensor.'}
        if missing:
            return {'value': None, 'unit': unit, 'stated': len(rows) - len(missing), 'missing': missing,
                    'reason': f'{what} is not stated for {", ".join(missing)}.'}
        return {'value': round(sum(float(r[key]) for r in rows), 6), 'unit': unit, 'stated': len(rows),
                'missing': [], 'reason': ''}

    cost = PR.total(rows)
    return {'variant': v.name, 'variant_uid': v.uid,
            'cameras': sum(1 for s in on if s.kind in CAMERA_KINDS),
            'sensors': len(on), 'tactile': sum(1 for s in on if s.kind == 'tactile'),
            'disabled': len(v.sensors) - len(on),
            'cost': cost, 'mass_g': total('mass_g', 'g', 'The mass'), 'power_w': total('power_w', 'W', 'The power'),
            'devices': rows,
            'notes': ['Only the enabled sensors of the active variant are counted; hosts, cables, lights and brackets '
                      'are not in the totals.',
                      'Prices are list prices on the date read (no tax, shipping or discount), each from the source '
                      'its library entry cites; no exchange rate is applied.']}


@jobs.study(STUDY_ID, 'Rig inventory', quick=False, batch='rig-inventory', needs_robot=False)
def run(project, ctx: jobs.Ctx) -> dict:
    """Camera and sensor counts, and cost, mass and power where the library states them (module docstring)."""
    ctx.report(1.0, 'Rig inventory')
    return inventory(project)
