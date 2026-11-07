"""Cited library prices (core/prices.py), the price block of library.schema.json, rig.cost on the samples, and the
prices' validation checks."""
import datetime as dt
import glob
import json
import os

import pytest

from dexvantage.core import library as LB
from dexvantage.core import metrics as M
from dexvantage.core import prices as PR
from dexvantage.core.project import Project, Sensor
from dexvantage.core.robot.samples import sample_project
from dexvantage.core.studies import rig_inventory as RI

TODAY = dt.date(2026, 11, 7)
GOOD = {'amount': 375, 'currency': 'USD', 'unit': 'each', 'kind': 'maker_store',
        'source': 'https://store.example.com/product', 'read': '2026-11-07', 'region': 'US', 'what': 'test figure'}


def _files():
    root = os.path.join(LB.RESOURCES, 'library')
    for f in sorted(glob.glob(os.path.join(root, '*', '*.json'))):
        with open(f, encoding='utf-8') as fh:
            yield f, json.load(fh)


def _entry_with(price):
    d = json.loads(json.dumps(LB.get('intel-realsense-d435').data))
    d['price'] = price
    return d


def _rig(*model_ids):
    p = Project('Price test')
    p.variant.sensors = [Sensor(name=f'S{k}', model_id=m) for k, m in enumerate(model_ids)]
    return p


def _cost(p):
    M.load_metrics()
    return M.read(M.get('rig.cost'), RI.inventory(p))


# ------------------------------------------------------------------ the schema
def test_schema_accepts_a_full_price_block_and_the_older_value_form():
    assert LB.validate(_entry_with(GOOD)) == []
    assert LB.validate(_entry_with(None)) == []
    assert LB.validate(_entry_with({'value': 90.0, 'currency': 'EUR'})) == []          # a custom entry's older form
    opts = dict(GOOD, amount=None, options=[{'label': '1 m', 'amount': 15, 'length_m': 1.0}])
    assert LB.validate(_entry_with(opts)) == []


@pytest.mark.parametrize('bad', [dict(GOOD, amount=0), dict(GOOD, amount=-3), dict(GOOD, currency='usd'),
                                 dict(GOOD, unit='pack'), dict(GOOD, kind='guess'), dict(GOOD, read='7 Nov 2026'),
                                 {'currency': 'USD'}, dict(GOOD, min_quantity=0)])
def test_schema_refuses_a_broken_price_block(bad):
    assert LB.validate(_entry_with(bad)), bad


def test_the_built_in_library_loads_with_prices():
    assert LB.problems() == []
    s = PR.summary()
    assert s['priced'] >= 60 and set(s['currencies']) == {'USD'}
    assert s['by_kind']['depth'][0] == s['by_kind']['depth'][1]          # every depth camera states a price


# ------------------------------------------------------------------ every price is cited
def test_every_built_in_price_has_a_url_a_date_a_kind_and_a_unit():
    n = 0
    for f, d in _files():
        p = d.get('price')
        if p is None:
            continue
        n += 1
        assert p['source'].startswith('https://'), f
        assert dt.date.fromisoformat(p['read']) <= TODAY + dt.timedelta(days=3650), f
        assert p['kind'] in PR.KINDS and p['unit'] == 'each' and p['currency'] in PR.ISO_4217, f
        assert isinstance(p.get('what'), str) and p['what'], f
        assert PR.problems(p, TODAY) == [], (f, PR.problems(p, TODAY))
        if p['kind'] == 'reseller':                    # said why: the maker sells only through it
            assert d['kind'] in ('tactile', 'robot', 'teleop'), f
    assert n >= 60


def test_volume_prices_say_their_quantity():
    for i in ('nvidia-jetson-orin-nx-16gb', 'nvidia-jetson-agx-orin-64gb', 'nvidia-jetson-t5000'):
        p = LB.get(i).data['price']
        assert p['kind'] == 'maker_price_list' and p['min_quantity'] == 1000 and '1KU+' in p['what']
    assert 'min_quantity' not in LB.get('nvidia-jetson-orin-nano-super-devkit').data['price']


def test_a_cable_sold_in_lengths_states_options_not_a_guessed_amount():
    p = LB.get('stereolabs-gmsl2-fakra-f-f').data['price']
    assert p['amount'] is None and [o['length_m'] for o in p['options']] == [0.3, 1.5, 5, 10]
    assert PR.stated(LB.get('stereolabs-gmsl2-fakra-f-f')) == (None, None)


# ------------------------------------------------------------------ problems()
def test_problems_find_each_broken_rule():
    assert PR.problems(GOOD, TODAY) == []
    assert any('positive' in m for m in PR.problems(dict(GOOD, amount=0), TODAY))
    assert any('ISO 4217' in m for m in PR.problems(dict(GOOD, currency='XYZ'), TODAY))
    assert any('stale' in m for m in PR.problems(dict(GOOD, read='2024-11-06'), TODAY))
    assert PR.problems(dict(GOOD, read='2024-11-08'), TODAY) == []         # 729 days: within the two years
    assert any('future' in m for m in PR.problems(dict(GOOD, read='2026-11-08'), TODAY))
    assert any('source' in m for m in PR.problems(dict(GOOD, source='the store'), TODAY))
    assert any('null' in m for m in PR.problems(dict(GOOD, amount=None), TODAY))


def test_stated_reads_amount_or_value_and_never_assumes_a_currency():
    assert PR.stated({'price': GOOD}) == (375.0, 'USD')
    assert PR.stated({'price': {'value': 10, 'currency': 'eur'}}) == (10.0, 'EUR')
    assert PR.stated({'price': {'amount': 10}}) == (None, None)
    assert PR.stated({'price': dict(GOOD, amount=True)}) == (None, None)
    assert PR.stated(None) == (None, None)


# ------------------------------------------------------------------ total() and rig.cost
def test_total_sum_lower_bound_not_stated_and_mixed_currencies():
    r = PR.total([{'name': 'a', 'price': 100.0, 'currency': 'USD'}, {'name': 'b', 'price': 50.5, 'currency': 'USD'}])
    assert r['value'] == 150.5 and r['bound'] is None and r['missing'] == [] and r['unit'] == 'USD'
    r = PR.total([{'name': 'a', 'price': 100.0, 'currency': 'USD'}, {'name': 'b', 'price': None, 'currency': None}])
    assert r['value'] == 100.0 and r['bound'] == 'at_least' and r['missing'] == ['b']
    assert 'At least 100.00 USD' in r['reason'] and '1 of the 2 parts' in r['reason']
    r = PR.total([{'name': 'b', 'price': None, 'currency': None}])
    assert r['value'] is None and 'not stated for any of the 1 parts' in r['reason']
    r = PR.total([{'name': 'a', 'price': 1.0, 'currency': 'USD'}, {'name': 'b', 'price': 1.0, 'currency': 'EUR'}])
    assert r['value'] is None and 'different currencies (EUR, USD)' in r['reason'] and 'exchange rate' in r['reason']
    assert PR.total([])['reason'] == 'The rig has no sensor.'


@pytest.mark.parametrize('name, want', [('ur5e', 325 + 375), ('aloha', 4 * 325), ('panda-bimanual', 2 * 325 + 375),
                                        ('g1', 375)])
def test_rig_cost_on_the_samples_whose_sensors_all_state_a_price(name, want):
    row = _cost(sample_project(name))
    assert row['status'] == 'ok' and row['value'] == pytest.approx(want) and 'bound' not in row


def test_rig_cost_is_a_lower_bound_on_the_dexhand_sample():
    p = sample_project('dexhand')                   # OVM6948 (no maker store price), See3CAM_24CUG, D435
    inv = RI.inventory(p)
    assert inv['cost']['missing'] == ['Palm micro camera (sample placement)'] and inv['cost']['stated'] == 2
    row = _cost(p)
    assert row['value'] == pytest.approx(199 + 375) and row['bound'] == 'at_least'
    assert '1 of the 3 parts' in row['reason']
    by = {d['model_id']: d for d in inv['devices']}
    assert by['e-con-see3cam-24cug']['price_source'].startswith('https://www.e-consystems.com/')
    assert by['e-con-see3cam-24cug']['price_read'] == '2026-11-07'


def test_rig_cost_is_not_stated_on_the_so100_sample():
    row = _cost(sample_project('so100'))            # sample placements with no library device
    assert row['value'] is None and row['status'] == 'not stated' and 'not stated for any' in row['reason']


def test_rig_cost_refuses_a_currency_other_than_usd():
    e = LB.get('intel-realsense-d405')
    saved = e.data
    try:
        e.data = dict(e.data, price=dict(GOOD, currency='EUR', amount=300))
        inv = RI.inventory(_rig('intel-realsense-d405'))
        assert inv['cost']['value'] == 300 and inv['cost']['unit'] == 'EUR'
        row = _cost(_rig('intel-realsense-d405'))
        assert row['value'] is None and 'exchange rate' in row['reason'] and 'EUR' in row['reason']
        row = _cost(_rig('intel-realsense-d405', 'intel-realsense-d435'))
        assert row['value'] is None and 'different currencies' in row['reason']
    finally:
        e.data = saved


def test_a_lower_bound_decides_a_cost_requirement_only_one_way():
    from dexvantage.core.requirements import decide
    assert decide(574.0, 'at_least', '<=', 500.0)[0] == 'fail'           # at least 574 cannot be at most 500
    assert decide(574.0, 'at_least', '<=', 1000.0)[0] == 'not_evaluated'  # the unpriced part might push it over


def test_optimiser_cost_objective_runs_when_every_candidate_states_a_price():
    from dexvantage.core.optimise import objectives as OB
    assert OB.price_stated('intel-realsense-d405') and not OB.price_stated('omnivision-ovm6948')
    assert not OB.price_stated('stereolabs-gmsl2-fakra-f-f')            # options only: no single amount


# ------------------------------------------------------------------ API, tool, validation
def test_api_prices_and_price():
    import dexvantage as dv
    t = dv.prices('depth')
    assert len(t) == 33 and all(r['amount'] and r['source'].startswith('https://') for r in t)
    assert dv.price('intel-realsense-d435')['amount'] == 375
    assert dv.price('omnivision-ovm6948') is None
    with pytest.raises(dv.DexVantageError):
        dv.price('no-such-device')
    every = dv.prices(stated_only=False)
    assert len(every) > len(dv.prices())


def test_assistant_tool_reads_the_rig_cost():
    from dexvantage.assistant import tools as T

    class _S:
        def project(self):
            return sample_project('ur5e')
    out = T.TOOLS['get_rig_cost'].fn(_S())
    assert out['cost']['value'] == pytest.approx(700) and out['library']['priced'] >= 60
    assert all(p['price_source'] for p in out['parts'])


def test_the_price_validation_checks_pass():
    from dexvantage.validation import suite
    suite.load_checks()
    specs = {s.id: s for s in suite.CHECKS}
    st = suite.ValidationSettings.from_dict({'quick': True, 'models': ['library']})
    for cid in ('library.prices_sound', 'library.rig_cost_sum'):
        c = suite.run_one(specs[cid], suite.Ctx(None, None, None, st))
        assert c.passed, (cid, c.exception, c.error, c.detail if hasattr(c, 'detail') else None)


def test_the_price_check_catches_a_stale_price():
    from dexvantage.validation import checks_prices as CP
    assert CP._rule_breaks(dict(GOOD, read='2020-01-01'), set(PR.ISO_4217), TODAY)
    assert CP._rule_breaks(dict(GOOD, currency='ABC'), set(PR.ISO_4217), TODAY)
    assert CP._rule_breaks(GOOD, set(PR.ISO_4217), TODAY) == []
