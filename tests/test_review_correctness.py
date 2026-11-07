"""Regression tests of the correctness review of research tools part 2: one test per defect found and fixed.

- Fisher's exact test named its one-sided alternatives the other way round from the rest of core/stats (the z test,
  the risk difference and the odds ratio are all condition 2 against condition 1).
- A short os.write in a shard's output file left half a line that the next line ran into (both lost).
- The shared disk cache never replaced a short or corrupt entry (every later get missed it).
- Two deletes of one run at once: the second raised a raw OSError instead of a sentence.
- A tolerance run's samples dropped the draws without a value, so two runs with one seed that each lost a different
  draw were paired off by one; the tracker's significance test refused lists holding a missing draw.
- Dataset pooling compared joint kinds by position after a reorder (a false "not poolable"), took a unit not stated
  as another kind, and ignored a target recorded in degrees.
- The optimiser kept the cost objective when a device is priced in another currency (rig.cost is then not stated
  for every rig holding it); a price without a currency counted as the currency "None".
"""
import json
import os

import numpy as np
import pytest

from dexvantage.core import cells_pool as CP
from dexvantage.core import prices as PR
from dexvantage.core import statcompare as SC
from dexvantage.core import stats as S
from dexvantage.core import tracking as TR
from dexvantage.core.cluster import cache as CA
from dexvantage.core.cluster import shards as SH
from dexvantage.core.tracking import record as REC
from dexvantage.core.tracking.store import RunStore, StoreError


# ====================================================================== statistics
@pytest.mark.parametrize('k1, n1, k2, n2', [(3, 10, 8, 10), (14, 20, 19, 20), (7, 9, 2, 13), (45, 100, 60, 100)])
def test_fisher_one_sided_is_b_against_a_like_the_z_test(k1, n1, k2, n2):
    from scipy import stats as ss
    for alt in ('less', 'greater'):
        f = S.fisher_exact(k1, n1, k2, n2, alt)['p']
        z = S.two_proportion_z(k1, n1, k2, n2, False, alt)['p']
        # both tests point the same way: small p together, large p together
        assert (f < 0.5) == (z < 0.5), (alt, f, z)
        ref = ss.fisher_exact([[k2, n2 - k2], [k1, n1 - k1]], alternative=alt).pvalue
        assert f == pytest.approx(ref, abs=1e-12)
    two = S.fisher_exact(k1, n1, k2, n2)['p']
    assert two == pytest.approx(ss.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]]).pvalue, abs=1e-12)


def test_fisher_greater_when_b_succeeds_more():
    # B (19 of 20) clearly above A (5 of 20): 'greater' (p2 > p1) is tiny, 'less' near 1
    assert S.fisher_exact(5, 20, 19, 20, 'greater')['p'] < 1e-4
    assert S.fisher_exact(5, 20, 19, 20, 'less')['p'] > 0.99
    r = S.compare_rates(5, 20, 19, 20, 'fisher', alternative='greater')
    assert r['p'] < 1e-4 and r['difference']['value'] > 0


# ====================================================================== cluster
def test_shard_line_survives_a_short_write(tmp_path, monkeypatch):
    path = str(tmp_path / 'shard-00000-of-00001.abc.jsonl')
    real = os.write

    def short(fd, data):                     # at most 7 bytes per call, as a slow or interrupted writer may
        return real(fd, bytes(data[:7]))
    monkeypatch.setattr(SH.os, 'write', short)
    SH.append_line(path, {'type': 'item', 'index': 0, 'status': 'ok', 'result': {'v': 1.5}})
    SH.append_line(path, {'type': 'item', 'index': 1, 'status': 'ok', 'result': {'v': 2.5}})
    recs = list(SH.read_lines(path))
    assert [r['index'] for r in recs] == [0, 1] and recs[1]['result'] == {'v': 2.5}


def test_disk_cache_replaces_a_corrupt_entry(tmp_path):
    c = CA.DiskResultCache(str(tmp_path / 'cache'))
    key = ('coverage', 'abc', 'def')
    p = c.path(key)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'wb') as fh:                # a file cut short (a full disk, a copy by hand)
        fh.write(b'{"schema": "dexvantage.cluster-cache/1"')
    assert CA.read_entry(p, key) is None
    c.put(key, {'value': 3.0})
    assert CA.read_entry(p, key) == {'value': 3.0}
    assert CA.DiskResultCache(str(tmp_path / 'cache')).get(key) == {'value': 3.0}


# ====================================================================== the tracker
def test_second_delete_of_a_run_is_a_sentence(tmp_path, monkeypatch):
    st = RunStore(str(tmp_path / 'runs'))
    rec = st.add({'study': 'coverage', 'started_utc': '2026-11-07T10:00:00.000Z', 'status': 'finished'})
    rid = rec['run_id']
    src = st.run_dir(rid)
    st.delete(rid)
    # a second writer passed the first check before the run went (the outer isdir said yes)
    real = os.path.isdir
    calls = {'n': 0}

    def isdir(p):
        if os.path.abspath(p) == os.path.abspath(src) and calls['n'] == 0:
            calls['n'] += 1
            return True
        return real(p)
    monkeypatch.setattr('dexvantage.core.tracking.store.os.path.isdir', isdir)
    with pytest.raises(StoreError, match='another writer deleted it'):
        st.delete(rid)


def test_tolerance_samples_keep_draw_order_with_gaps():
    out, samples = {}, {}
    doc = {'done': 5, 'metrics': {'coverage.covered': {'title': 'Seen', 'unit': '%', 'better': 'higher',
                                                       'values': [90.0, None, 91.0, float('nan'), 92.0]}}}
    REC._tolerance(doc, out, samples)                                      # noqa: SLF001
    assert samples['coverage.covered'] == [90.0, None, 91.0, None, 92.0]
    json.dumps(samples, allow_nan=False)


def test_significance_drops_missing_pairs_not_a_shifted_list():
    rng = np.random.default_rng(1)
    a = list(rng.normal(90, 1, 20))
    b = [x + 0.5 for x in a]
    a[3], b[11] = None, None                 # each run lost a different draw
    r = TR.significance(a, b, 'higher', paired=True)
    assert r.get('available') and r['paired'] and r['a']['n'] == 18
    assert r['difference']['value'] == pytest.approx(0.5, abs=1e-9)        # pairs aligned: every difference 0.5
    u = TR.significance(a, b, 'higher', paired=False)
    assert u.get('available') and u['a']['n'] == 19 and u['b']['n'] == 19


class _Store:
    def __init__(self, recs):
        self.recs = {r['run_id']: r for r in recs}

    def runs(self, **_):
        return list(self.recs.values())

    def get(self, rid):
        return self.recs[rid]


def _rec(rid, samples):
    return {'run_id': rid, 'name': rid, 'study': 'tolerance', 'seeds': {'tolerance': 7}, 'tags': [],
            'metrics': {'coverage.covered': {'value': 90.0, 'title': 'Seen', 'unit': '%', 'better': 'higher'}},
            'samples': {'coverage.covered': samples}}


def test_tracker_samples_pair_by_draw_across_gaps():
    a = [90.0 + 0.1 * i for i in range(12)]
    b = [x + 1.0 for x in a]
    a[2], b[7] = None, None
    data = SC.tracker_samples(_Store([_rec('x' * 32, a), _rec('y' * 32, b)]), 'x' * 32, 'y' * 32)
    assert data['paired']
    res = SC.analyse_samples(data['rows'], data['paired'], n_boot=500, n_perm=500)
    c = res['comparisons'][0]
    assert c['dropped'] == 2 and c['differences']['n'] == 10
    assert c['differences']['sd'] == pytest.approx(0.0, abs=1e-9)          # every pair differs by exactly 1


# ====================================================================== dataset pooling
def _sc(**kw):
    base = {'source': 'plan', 'fps': {'value': 10.0}, 'control_hz': {'value': 10.0}, 'cameras': [],
            'state': {'names': ['arm', 'slide'], 'count': 2, 'units': ['rad', 'm']},
            'action': {'names': ['arm', 'slide'], 'count': 2, 'units': ['rad', 'm'],
                       'space': {'value': 'joint position'}},
            'gripper': {'joints': [], 'type': None, 'range': None}}
    base.update(kw)
    return base


def test_pooling_compares_units_after_the_reorder():
    t = _sc()
    cell = _sc(state={'names': ['slide', 'arm'], 'count': 2, 'units': ['m', 'rad']},
               action={'names': ['slide', 'arm'], 'count': 2, 'units': ['m', 'rad'],
                       'space': {'value': 'joint position'}})
    r = CP.rules(t, cell)
    assert r['verdict'] == 'poolable after mapping', r['unmappable']
    assert [o['op'] for o in r['operations']] == ['reorder', 'reorder']
    # a real kind mismatch after the reorder is still caught
    bad = _sc(state={'names': ['slide', 'arm'], 'count': 2, 'units': ['rad', 'rad']})
    assert CP.rules(t, bad)['verdict'] == 'not poolable'


def test_pooling_unit_not_stated_is_not_another_kind():
    r = CP.rules(_sc(), _sc(state={'names': ['arm', 'slide'], 'count': 2, 'units': ['rad', None]}))
    assert r['verdict'] == 'poolable as is', r['unmappable']


def test_pooling_target_recorded_in_degrees_converts_the_cell():
    hint = {'value': 'deg', 'joints': ['arm'], 'source': 'stored values reach 170.0, beyond ±2π'}
    t = _sc(state={'names': ['arm', 'slide'], 'count': 2, 'units': None, 'units_hint': hint})
    r = CP.rules(t, _sc())
    conv = [o for o in r['operations'] if o['op'] == 'convert']
    assert len(conv) == 2 and conv[0]['from'] == 'rad' and conv[0]['to'] == 'deg' and conv[0]['joints'] == ['arm']
    both = CP.rules(t, _sc(state={'names': ['arm', 'slide'], 'count': 2, 'units': None, 'units_hint': hint}))
    assert not any(o['op'] == 'convert' for o in both['operations'])
    assert any('degrees' in w for w in both['warnings'])


# ====================================================================== prices
def test_cost_objective_needs_a_usd_price(monkeypatch):
    from dexvantage.core import library as LB
    from dexvantage.core.optimise import objectives as OB

    class _E:
        def __init__(self, cur):
            self.data = {'price': {'amount': 100.0, 'currency': cur}}
    monkeypatch.setattr(LB, 'get', lambda mid: _E('EUR' if mid == 'eur-cam' else 'USD'))
    assert OB.price_stated('usd-cam') and not OB.price_stated('eur-cam')


def test_total_counts_a_price_without_currency_as_missing():
    r = PR.total([{'name': 'a', 'price': 10.0, 'currency': 'USD'}, {'name': 'b', 'price': 5.0, 'currency': None}])
    assert r['value'] == 10.0 and r['bound'] == 'at_least' and r['missing'] == ['b'] and r['currencies'] == ['USD']


# ====================================================================== campaign storage
def test_campaign_says_a_compression_ratio_above_one_grows_the_recording():
    from dexvantage.core import campaign as CPN
    from dexvantage.core.project import blank_project
    spec = {'tasks': [{'name': 'Pick', 'episode_s': 20}], 'demos': {'per_condition': 4}, 'format': 'lerobot'}
    doc = CPN.plan(blank_project('t'), {**spec, 'compression_ratio': {'value': 20, 'source': 'pilot'}})
    assert any('encoded size over the raw size' in n and '0.05' in n for n in doc['notes'])
    ok = CPN.plan(blank_project('t'), {**spec, 'compression_ratio': {'value': 0.05, 'source': 'pilot'}})
    assert not any('encoded size over the raw size' in n for n in ok['notes'])
