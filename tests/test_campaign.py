"""The data-collection campaign planner (core/campaign.py, the study 'campaign', the exporter 'campaign', batch
--campaign, dv.campaign, the metrics and the assistant's get_campaign): the design's balance, the limits, storage
against the bandwidth study, the power analysis, the exports read back, the refusals in one sentence."""
import csv
import datetime as dt
import io
import json
import os
import subprocess
import sys

import pytest

from dexvantage.core import campaign as CP
from dexvantage.core import jobs
from dexvantage.core.project import Variant, blank_project
from dexvantage.core.robot.samples import sample_project
from dexvantage.validation import checks_campaign as CC
from dexvantage.validation import suite as VS

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SPEC2 = {'operators': ['Ana', 'Ben'], 'lighting': [{'name': 'Bright'}, {'name': 'Dim'}],
         'tasks': [{'name': 'Pick', 'episode_s': 20}, {'name': 'Place', 'episode_s': 30}],
         'demos': {'per_condition': 12}, 'start_date': '2026-11-09'}


def _check(cid, quick=True):
    spec = next(s for s in VS.CHECKS if s.id == cid)
    ctx = VS.Ctx(None, None, None, VS.ValidationSettings(quick=quick))
    try:
        return VS.run_one(spec, ctx)
    finally:
        ctx.close()


@pytest.fixture(scope='module')
def ur5e():
    return sample_project('ur5e')


# ====================================================================== the design
def test_williams_rows_and_the_check():
    assert CP.williams(4)[0] == [0, 1, 3, 2]
    assert len(CP.williams(3)) == 6 and len(CP.williams(1)) == 1 and CP.williams(0) == []
    c = _check('campaign.williams_square')
    assert c.passed and c.error == 0, c


def test_balance_and_limits_on_random_campaigns():
    for cid in ('campaign.schedule_balance', 'campaign.session_limits'):
        c = _check(cid, quick=False)
        assert c.passed and c.error == 0, (cid, c.exception, c.detail)
    assert _check('campaign.schedule_balance', quick=False).detail['carryover_exact'] > 10


def test_two_by_two_with_two_operators_is_fully_balanced():
    doc = CP.plan(blank_project('t'), SPEC2)
    d = doc['design']
    assert d['conditions'] == 4 and d['square_rows'] == 4 and (d['sequences'] % 4) == 0
    assert d['position_balance'] == d['carryover_balance'] == 'exact'
    b = CP.balance(doc)
    assert b['operator_equal'] and b['position_equal'] and b['carryover_equal']
    assert {c['recorded'] for c in doc['conditions']} == {12}
    assert sum(s['episodes'] for s in doc['sessions']) == 48 == doc['totals']['episodes']
    assert doc['headline']['text'].startswith('48 episodes over ')
    json.dumps(doc, allow_nan=False)


def test_partial_balance_is_said():
    doc = CP.plan(blank_project('t'), {'tasks': [{'name': x, 'episode_s': 10} for x in 'ABC'],
                                       'demos': {'per_condition': 1}})
    d = doc['design']
    assert d['rounds'] == 1 and d['position_balance'] == 'partial' and 'partial' in d['text']
    assert CP.balance(doc)['operator_equal']


def test_a_tampered_schedule_fails_the_balance_count():
    doc = CP.plan(blank_project('t'), SPEC2)
    rows = list(csv.DictReader(io.StringIO(CP.to_csv(doc))))
    rows[0]['episodes'] = str(int(rows[0]['episodes']) + 1)
    per = {}
    for r in rows:
        per[(r['operator'], r['condition'])] = per.get((r['operator'], r['condition']), 0) + int(r['episodes'])
    assert len(set(per.values())) > 1                 # the count the check makes would see it


def test_seed_changes_the_order_but_not_the_balance():
    a = CP.plan(blank_project('t'), {**SPEC2, 'seed': 1})
    b = CP.plan(blank_project('t'), {**SPEC2, 'seed': 2})
    assert [s['order'] for s in a['sequences']] != [s['order'] for s in b['sequences']]
    assert CP.balance(b)['carryover_equal'] and CP.plan(blank_project('t'), {**SPEC2, 'seed': 1})['sequences'] \
        == a['sequences']


def test_keep_rate_records_enough():
    doc = CP.plan(blank_project('t'), {**SPEC2, 'keep_rate': 0.8})
    for c in doc['conditions']:
        assert c['recorded'] * 0.8 >= 12 and c['expected_kept'] == pytest.approx(c['recorded'] * 0.8)
    assert any('more than' in n for n in doc['notes']) or doc['conditions'][0]['recorded'] == 15


def test_operators_rotate_and_days_keep_limits():
    sp = {**SPEC2, 'session_max_min': 15, 'sessions_per_day': 2, 'operator_sessions_per_day': 1}
    doc = CP.plan(blank_project('t'), sp)
    first = [next(s['operator'] for s in doc['sessions'] if s['day'] == d['day']) for d in doc['days']]
    assert first[:2] == ['Ana', 'Ben'] or first[:2] == ['Ben', 'Ana']
    assert first[0] != first[1]
    for d in doc['days']:
        ss = [s for s in doc['sessions'] if s['day'] == d['day']]
        assert len(ss) <= 2 and len({s['operator'] for s in ss}) == len(ss)
    assert all(s['minutes'] <= 15 for s in doc['sessions'])


def test_weekdays_only_skips_the_weekend():
    sp = {**SPEC2, 'start_date': '2026-11-13', 'sessions_per_day': 1, 'session_max_min': 30}
    doc = CP.plan(blank_project('t'), sp)
    dates = [d['date'] for d in doc['days']]
    assert dates[:3] == ['2026-11-13', '2026-11-16', '2026-11-17']
    doc = CP.plan(blank_project('t'), {**sp, 'weekdays_only': False})
    assert [d['date'] for d in doc['days']][:2] == ['2026-11-13', '2026-11-14']


# ====================================================================== inputs from the project
def test_defaults_are_labelled_and_the_bimanual_task_is_one(ur5e):
    p = sample_project('aloha')
    doc = jobs.run_study('campaign', p)
    assert [t['name'] for t in doc['levels']['task']] == ['Handover']
    c = doc['conditions'][0]
    assert c['episode_s'] == pytest.approx(4.0) and 'trajectories' in c['episode_source']
    for k in ('operators', 'reset_s', 'session_max_min', 'demos.per_condition', 'format'):
        assert k in doc['defaults'], k
    assert doc['headline']['status_word'] == 'no budget set' and doc['headline']['status'] == 'warning'
    b = blank_project('t')
    d2 = CP.plan(b, {})
    assert d2['conditions'][0]['episode_default'] and 'a default' in d2['conditions'][0]['episode_source']


def test_checklist_names_calibration_lighting_exposure_and_devices(ur5e):
    p = ur5e.snapshot()
    p.workspace.lights.append({'uid': 'l1aaaaaaaa', 'name': 'Key panel', 'kind': 'panel', 'pose': {
        'position': [0.5, 0, 1.5], 'quaternion': [0, 0, 0, 1]}, 'size_m': [0.6, 0.6]})
    s = p.variant.sensors[0]
    p.analysis.studies['motion_blur'] = {'exposure_s': {s.uid: {'value': 0.002, 'source': 'Apply'}}}
    spec = {'lighting': [{'name': 'Full', 'lights': {'Key panel': 1.0}}, {'name': 'Off', 'lights': {'Key panel': 0}}]}
    doc = CP.plan(p, spec)
    first = doc['sessions'][0]['checklist']
    items = {c['item'] for c in first}
    assert {'calibration', 'lighting', 'exposure', 'device', 'objects', 'disk', 'episodes', 'rig'} <= items
    texts = ' '.join(c['text'] for c in first)
    assert 'Key panel off' in texts or 'Key panel on at 100 %' in texts
    assert f'{s.name}: exposure 2 ms (motion blur: Apply)' in texts
    assert 'not set' in texts                         # the other camera
    assert '0.5 mm' in texts and '(defaults)' in texts
    assert sum(c['calibration'] for c in doc['sessions'] if c['day'] == 1) == 1


def test_storage_matches_the_bandwidth_study_and_formats(ur5e):
    c = _check('campaign.storage_vs_bandwidth')
    assert c.passed, (c.exception, c.detail)
    raw = CP.plan(ur5e, {'episode_s': 10})
    lr = CP.plan(ur5e, {'episode_s': 10, 'format': 'lerobot'})
    assert lr['totals']['storage_gb'] == raw['totals']['storage_gb']
    assert 'no source states' in lr['storage'][ur5e.variant.name]['ratio_source']
    lr2 = CP.plan(ur5e, {'episode_s': 10, 'format': 'lerobot', 'compression_ratio': {'value': 0.1, 'source': 'x'}})
    assert lr2['totals']['storage_gb'] == pytest.approx(raw['totals']['storage_gb'] * 0.1, rel=1e-3)


def test_a_sensor_without_a_rate_makes_storage_a_lower_bound():
    from dexvantage.core import metrics as M
    p = sample_project('dexhand')
    p.analysis.studies['campaign'] = {'budget': {'disk_gb': 1e6}}
    doc = jobs.run_study('campaign', p)
    assert doc['totals']['storage_bound'] == 'at_least' and doc['totals']['storage_incomplete']
    assert doc['headline']['status_word'] == 'storage incomplete'
    ev = M.evaluate(p, ['campaign.storage_gb', 'campaign.episodes', 'campaign.days', 'campaign.kept_per_condition'])
    assert ev['campaign.storage_gb']['bound'] == 'at_least'
    assert ev['campaign.episodes']['value'] == doc['totals']['episodes']
    assert ev['campaign.kept_per_condition']['value'] == 50


def test_variants_as_conditions_store_their_own_rates(ur5e):
    p = ur5e.snapshot()
    v = Variant.from_dict(p.variant.to_dict())
    v.uid, v.name = 'b0b0b0b0b0', 'Without the overhead camera'
    v.sensors = [s for s in v.sensors if 'Overhead' not in s.name]
    p.variants.append(v)
    doc = CP.plan(p, {'variants': [p.variants[0].name, v.name], 'episode_s': 10})
    by = {c['variant']: c['gb_per_episode'] for c in doc['conditions']}
    assert by[v.name] < by[p.variants[0].name] and len(doc['storage']) == 2


# ====================================================================== the budget
def test_budget_status_words(ur5e):
    ok = CP.plan(ur5e, {'budget': {'disk_gb': 1e6, 'days': 100}})
    assert ok['budget']['status'] == 'good' and ok['headline']['status_word'] == 'within budget'
    over = CP.plan(ur5e, {'budget': {'disk_gb': 10, 'days': 100}})
    assert over['headline']['status'] == 'critical' and over['headline']['status_word'] == 'over budget'
    assert any('compressed format' in a for a in over['advice'])
    gb = over['totals']['storage_gb']
    tight = CP.plan(ur5e, {'budget': {'disk_gb': gb / 0.95}})
    assert tight['headline']['status'] == 'warning' and tight['headline']['status_word'] == 'tight'
    up = CP.plan(ur5e, {'budget': {'upload_mbps': 1, 'upload_hours_per_day': 2}})
    assert up['budget']['checks'][0]['status'] == 'critical' and up['totals']['upload_hours_total'] > 2


# ====================================================================== power analysis
def test_power_analysis_against_the_exact_power():
    r = CP.power_n(0.5, 0.7, 0.05, 0.8, continuity=False)
    assert CC.exact_power(r['n'], 0.5, 0.7, 0.05, False) >= 0.79
    assert CC.exact_power(r['n'] - 10, 0.5, 0.7, 0.05, False) < 0.8
    assert CP.power_n(0.5, 0.7)['n'] > r['n']                     # the continuity correction asks for more
    for cc in (False, True):                                      # core/stats.py (when there) and the formula agree
        q = CP.power_n(0.6, 0.85, 0.05, 0.9, cc)
        assert q.get('formula_n', q['n']) == q['n']
    c = _check('campaign.power_exact', quick=False)
    assert c.passed, c.detail
    doc = CP.plan(blank_project('t'), {'demos': {'mode': 'power', 'p_base': 0.5, 'p_target': 0.7}})
    assert doc['demos']['target'] == CP.power_n(0.5, 0.7)['n'] and doc['demos']['power']['method']
    with pytest.raises(CP.CampaignError, match='two different success rates'):
        CP.power_n(0.5, 0.5)


# ====================================================================== refusals
@pytest.mark.parametrize('spec, words', [
    ({'variants': ['Nope']}, 'rig variant "Nope"'),
    ({'tasks': [{'name': 'Long', 'episode_s': 4000}], 'session_max_min': 30}, 'does not fit in a 30 min session'),
    ({'tasks': [{'name': f'T{i}'} for i in range(30)], 'lighting': [{'name': f'L{i}'} for i in range(30)]},
     'more than 500'),
    ({'lighting': [{'name': 'A'}, {'name': 'A'}]}, 'share a name'),
    ({'lighting': [{'name': 'A', 'lights': {'missing': 1}}]}, 'names a light "missing"'),
    ({'conditions': [{'lighting': 'Nope'}]}, 'not among the levels'),
    ({'tasks': [{'name': 'A', 'trajectory': 'zz'}]}, 'trajectory "zz"'),
    ({'session_max_min': 600, 'day_hours': 1, 'tasks': [{'name': 'A', 'episode_s': 4000}]}, 'fits in a 1 h day'),
])
def test_refusals_are_one_sentence(spec, words):
    with pytest.raises(CP.CampaignError) as e:
        CP.plan(blank_project('t'), spec)
    assert words in str(e.value) and '\n' not in str(e.value)


def test_spec_files_are_read_or_refused(tmp_path):
    f = tmp_path / 'spec.json'
    f.write_text(json.dumps({'schema': CP.SPEC_SCHEMA, 'operators': 2}))
    assert CP.settings_of(blank_project('t'), CP.read_spec(str(f)))['operators'] == ['Operator 1', 'Operator 2']
    for text, words in (('{"schema": "x/1"}', 'not a campaign spec'), ('[1]', 'must be a JSON object'),
                        ('{bad', 'not valid JSON')):
        with pytest.raises(CP.CampaignError, match=words):
            CP.read_spec(text)
    with pytest.raises(CP.CampaignError, match='Could not read'):
        CP.read_spec(str(tmp_path / 'missing.json'))


# ====================================================================== exports
def test_ics_reads_back_strictly(tmp_path):
    c = _check('campaign.ics_roundtrip', quick=False)
    assert c.passed and c.detail['events'] > 0, (c.exception, c.detail)
    doc = CP.plan(blank_project('Cell, "A"; B'), {**SPEC2, 'name': 'Pick, place; «ü» \\ test'})
    stamp = dt.datetime(2026, 11, 7, 9, 0, tzinfo=dt.timezone.utc)
    data = CP.to_ics(doc, dtstamp=stamp).encode('utf-8')
    got = CC.parse_ics(data)
    want = CP.ics_events(doc)
    assert [(e['UID'], e['DTSTART'], e['DTEND'], e['SUMMARY'], e['DESCRIPTION']) for e in got['events']] == \
        [(e['uid'], e['start'], e['end'], e['summary'], e['description']) for e in want]
    assert got['events'][0]['DTSTART'] == dt.datetime(2026, 11, 9, 9, 10)     # 09:00 + the 10 min daily check
    assert all(len(ln) <= 75 for ln in data.split(b'\r\n'))
    # the reader is strict: it refuses what RFC 5545 does not allow
    bad = [data.replace(b'\r\n', b'\n'), data.replace(b'UID:', b'X-UID:', 1),
           data.replace(b'DTSTAMP:20261107T090000Z', b'DTSTAMP:20261107T090000', 1),
           data.replace(b'SUMMARY:', b'SUMMARY:a,b ', 1), data.replace(b'END:VEVENT\r\n', b'', 1),
           data.replace(b'VERSION:2.0\r\n', b'', 1), b'BEGIN:VCALENDAR\r\nX:' + b'y' * 80 + b'\r\nEND:VCALENDAR\r\n']
    for b in bad:
        with pytest.raises(CC.IcsError):
            CC.parse_ics(b)


def test_csv_html_and_files(tmp_path):
    doc = CP.plan(blank_project('t'), {**SPEC2, 'name': '<script>x</script>'})
    rows = list(csv.DictReader(io.StringIO(CP.to_csv(doc))))
    assert sum(int(r['episodes']) for r in rows) == doc['totals']['episodes']
    assert len(rows) == sum(len(s['blocks']) for s in doc['sessions'])
    page = CP.to_html(doc)
    assert '<script>x' not in page and '&lt;script&gt;' in page and '@media print' in page
    assert doc['headline']['text'] in page and 'Williams' in page
    paths = CP.write_files(doc, str(tmp_path / 'plan'))
    assert [os.path.basename(p) for p in paths] == ['plan.csv', 'plan.ics', 'plan.html']
    assert open(paths[1], 'rb').read().count(b'\r\n') > 10


def test_ics_without_a_start_date_counts_from_the_given_one():
    doc = CP.plan(blank_project('t'), {k: v for k, v in SPEC2.items() if k != 'start_date'})
    assert doc['sessions'][0]['date'] is None
    ev = CP.ics_events(doc, '2026-12-01')
    assert ev[0]['start'].date() == dt.date(2026, 12, 1)


def test_exporter_writes_four_files_with_a_manifest(tmp_path, ur5e):
    from dexvantage.core import exporters as X
    man = X.export(ur5e, str(tmp_path), ['campaign'], {'campaign': {'spec': {'operators': ['A', 'B']},
                                                                    'start_date': '2026-11-09'}})
    row = man['formats']['campaign']
    assert row['status'] == 'ok', row
    assert sorted(row['files']) == ['campaign/plan.csv', 'campaign/plan.html', 'campaign/plan.ics',
                                    'campaign/plan.json']
    assert all(len(man['files'][f]) == 64 for f in row['files'])
    doc = json.load(open(tmp_path / 'campaign' / 'plan.json'))
    assert doc['operators'] == ['A', 'B'] and 'provenance' in doc
    CC.parse_ics((tmp_path / 'campaign' / 'plan.ics').read_bytes())


# ====================================================================== apply, API, assistant
def test_apply_writes_the_plan_and_the_study_ignores_it(ur5e):
    p = ur5e.snapshot()
    p.analysis.studies['campaign'] = {'operators': ['A', 'B']}
    doc = jobs.run_study('campaign', p)
    CP.apply_plan(p, doc)
    ap = CP.applied_plan(p)
    assert ap['digest'] == CP.plan_digest(doc) and ap['totals']['episodes'] == doc['totals']['episodes']
    assert p.analysis.studies['campaign']['operators'] == ['A', 'B']
    again = jobs.run_study('campaign', p)
    assert CP.plan_digest(again) == CP.plan_digest(doc)
    q = type(p).from_dict(json.loads(json.dumps(p.to_dict())))       # it survives the file
    assert CP.applied_plan(q)['digest'] == ap['digest']


def test_python_api():
    import dexvantage as dv
    rig = dv.sample('so100')
    res = dv.campaign.plan(rig, operators=['A', 'B'], demos={'per_condition': 6})
    assert res.study == 'campaign' and res.headline['metric'] == 'campaign.episodes'
    assert res.headline['value'] == res['totals']['episodes'] and res.headline['word'] == 'no budget set'
    assert 'campaign' not in rig.project.analysis.studies             # the rig is unchanged
    assert dv.campaign.balance(res)['operator_equal']
    assert dv.campaign.power_n(0.5, 0.8)['n'] > 0
    dv.campaign.apply(rig, res)
    assert CP.applied_plan(rig.project)['digest'] == CP.plan_digest(res)
    assert rig.project.analysis.studies['campaign']['operators'] == ['A', 'B']


def test_assistant_reads_the_campaign(ur5e):
    from dexvantage.assistant.session import FileSession
    from dexvantage.assistant.tools import call
    out = call(FileSession(project=ur5e.snapshot()), 'get_campaign', {})
    assert 'error' not in out and out['study'] == 'campaign'


# ====================================================================== batch
def _run(*args, cwd=None):
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    return subprocess.run([sys.executable, '-m', 'dexvantage', *args], capture_output=True, text=True, env=env,
                          cwd=cwd or ROOT, timeout=180)


def test_batch_campaign_spec(tmp_path):
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({'schema': CP.SPEC_SCHEMA, **SPEC2, 'budget': {'disk_gb': 1e6}}))
    r = _run('--sample', 'ur5e', '--campaign', str(spec))
    assert r.returncode == 0, r.stderr
    assert 'within budget' in r.stdout
    for ext in ('json', 'csv', 'ics', 'html'):
        assert (tmp_path / f'spec.plan.{ext}').is_file(), ext
    doc = json.loads((tmp_path / 'spec.plan.json').read_text())
    assert doc['study'] == 'campaign' and doc['provenance'] and str(tmp_path) not in json.dumps(doc)
    CC.parse_ics((tmp_path / 'spec.plan.ics').read_bytes())
    spec.write_text(json.dumps({**SPEC2, 'budget': {'disk_gb': 1}}))
    r = _run('--sample', 'ur5e', '--campaign', str(spec), '--campaign-out', str(tmp_path / 'over.json'))
    assert r.returncode == 1 and 'over budget' in r.stdout and (tmp_path / 'over.ics').is_file()
    spec.write_text(json.dumps({'variants': ['Nope']}))
    r = _run('--sample', 'ur5e', '--campaign', str(spec))
    assert r.returncode == 2 and 'Traceback' not in r.stderr and 'rig variant "Nope"' in r.stderr
    r = _run('--sample', 'ur5e', '--campaign-saved', str(tmp_path / 'saved.json'))
    assert r.returncode == 0, r.stderr
    assert json.loads((tmp_path / 'saved.json').read_text())['study'] == 'campaign'
    r = _run('--sample', 'ur5e', '--campaign-out', str(tmp_path / 'x.json'))
    assert r.returncode == 2 and 'goes with --campaign' in r.stderr
