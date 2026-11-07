"""Rig inventory (spec §7.11, §11): what the active variant's rig is made of, as the figures the optimiser and the
requirements scorecard compare: the camera count, the sensor count, and the cost, mass and power of the enabled
sensors where the library states them.

    @jobs.study('rig', 'Rig inventory', quick=False, batch='rig-inventory')   → --rig-inventory OUT.json

Each total is {'value', 'unit', 'stated', 'missing', 'reason'}: the sum over the enabled sensors when every one states
the figure, else value None ("not stated") with the devices that do not state it. The cost reads an optional "price"
object of a library entry ({"value": number, "currency": "USD", "source": ..., "read": "YYYY-MM-DD"}); prices in
different currencies are not added (no exchange rate is assumed).

Known limits: only the sensors are counted (hosts, hubs, cables, lights and brackets are not in the totals); a sensor
with no library device (the default camera) states nothing; no built-in library entry states a price today, so the
cost reads "not stated" until a custom entry gives one; mass and power are the maker's figures for the device alone
(no bracket, no cable).
"""
from __future__ import annotations

from .. import jobs
from .. import library as LB

STUDY_ID = 'rig'
CAMERA_KINDS = ('camera', 'depth')


def stated_price(entry) -> tuple:
    """(value, currency) of a library entry's stated price, or (None, None): the one rule for whether a price is
    stated (the rig inventory and the optimiser's cost objective use it)."""
    p = entry.data.get('price') if entry is not None else None
    if isinstance(p, dict) and isinstance(p.get('value'), (int, float)) and not isinstance(p.get('value'), bool):
        return float(p['value']), str(p.get('currency') or 'USD')
    return None, None


def inventory(project) -> dict:
    v = project.variant
    on = [s for s in v.sensors if s.enabled]
    rows = []
    for s in on:
        e = LB.get(s.model_id) if s.model_id else None
        price, cur = stated_price(e)
        rows.append({'uid': s.uid, 'name': s.name, 'kind': s.kind, 'model_id': s.model_id or '',
                     'device': e.title if e is not None else None,
                     'price': price, 'currency': cur,
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

    cost = total('price', 'USD', 'A price')
    curs = sorted({r['currency'] for r in rows if r['currency']})
    if cost['value'] is not None and len(curs) > 1:
        cost = {'value': None, 'unit': '', 'stated': len(rows), 'missing': [],
                'reason': f'The prices are in different currencies ({", ".join(curs)}): no exchange rate is assumed.'}
    elif curs:
        cost['unit'] = curs[0]
    return {'variant': v.name, 'variant_uid': v.uid,
            'cameras': sum(1 for s in on if s.kind in CAMERA_KINDS),
            'sensors': len(on), 'tactile': sum(1 for s in on if s.kind == 'tactile'),
            'disabled': len(v.sensors) - len(on),
            'cost': cost, 'mass_g': total('mass_g', 'g', 'The mass'), 'power_w': total('power_w', 'W', 'The power'),
            'devices': rows,
            'notes': ['Only the enabled sensors of the active variant are counted; hosts, cables, lights and brackets '
                      'are not in the totals.']}


@jobs.study(STUDY_ID, 'Rig inventory', quick=False, batch='rig-inventory', needs_robot=False)
def run(project, ctx: jobs.Ctx) -> dict:
    """Camera and sensor counts, and cost, mass and power where the library states them (module docstring)."""
    ctx.report(1.0, 'Rig inventory')
    return inventory(project)
