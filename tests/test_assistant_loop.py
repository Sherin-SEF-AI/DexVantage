"""The assistant's loop, headless, with the fake provider only (no network, no key, no real keyring): a tool-calling
turn, the refusal to invent numbers (the system prompt rule and the number post-check), the Copilot's approval gate
for long studies, edit proposals, the keyring-only keys, the chats and memory kept per project, plain-English errors.
"""
import json
import os

import pytest

from dexvantage.assistant import agent as A
from dexvantage.assistant import chats as CH
from dexvantage.assistant import copilot as CP
from dexvantage.assistant import edits as ED
from dexvantage.assistant import memory as MM
from dexvantage.assistant import numcheck as NC
from dexvantage.assistant import providers as PR
from dexvantage.assistant import routing as RT
from dexvantage.assistant import secrets as SK
from dexvantage.assistant import store as ST
from dexvantage.assistant import toolbox as TB
from dexvantage.assistant.client import ApiError
from dexvantage.assistant.fake import FakeReply
from dexvantage.core.robot.samples import sample_project

WRIST = 'Wrist camera (sample placement)'


@pytest.fixture(scope='module')
def so100():
    return sample_project('so100')


def turn(project, script, text='question', page=None, memory=''):
    """Run one scripted turn headless: (events, tool context, fake session)."""
    tc = TB.ToolContext(project_name=project.name, variant_name=project.variant.name, page_context=page,
                        memory=memory, provider=PR.FAKE, model='fake-model')
    tc.session = TB.file_session(project.snapshot(), tc)
    s = A.new_session(PR.FAKE, None, A.Settings('fake-model'), script=[script])
    ev = []
    s.run_turn(text, tc, ev.append)
    return ev, tc, s


def kinds(ev):
    return [e.kind for e in ev if e.kind not in ('content', 'usage', 'turn')]


def answer(ev):
    return next(e.data for e in ev if e.kind == 'answer')


# ======================================================================= a tool-calling turn
def test_a_tool_calling_turn_runs_the_tools_and_feeds_their_results_back(so100):
    def final(results):
        name, doc = results[0]
        return f'The project has {len(doc["robots"])} robot and the rig is "{doc["variant"]}".'
    ev, tc, s = turn(so100, [FakeReply(calls=[('get_overview', {})]), FakeReply(text=final)])
    assert kinds(ev) == ['tool_start', 'tool_done', 'answer', 'done']
    done = next(e for e in ev if e.kind == 'tool_done')
    assert done.data['ok'] and done.data['name'] == 'get_overview'
    doc = json.loads(done.data['result'])
    assert doc['project'] == so100.name or so100.name in json.dumps(doc)
    # the second request carried the tool result back to the model
    assert len(s.requests) == 2
    assert any(m['role'] == 'tool' and m['name'] == 'get_overview' for m in s.requests[1]['messages'])
    assert answer(ev)['check']['ok']
    assert ev[-1].data['ok'] is True
    # every tool the dock offers is listed to the model, the dock's own among them
    offered = set(s.requests[0]['tools'])
    assert {'get_overview', 'get_page_context', 'propose_changes', 'propose_plan', 'copilot_catalogue',
            'remember'} <= offered


def test_bad_tool_input_comes_back_as_a_sentence_and_the_turn_goes_on(so100):
    ev, _, _ = turn(so100, [FakeReply(calls=[('get_sensor', {'nonsense': 1}), ('no_such_tool', {})]),
                            FakeReply(text='I could not read that sensor.')])
    res = [json.loads(e.data['result']) for e in ev if e.kind == 'tool_done']
    assert all(not e.data['ok'] for e in ev if e.kind == 'tool_done')
    assert all('error' in r and 'Traceback' not in r['error'] for r in res)
    assert ev[-1].data['ok']


# ======================================================================= no invented numbers
def test_the_system_prompt_forbids_inventing_numbers():
    s = A.SYSTEM.lower()
    assert 'never guess' in s and 'invent' in s and 'not stated' in s
    assert 'get_page_context' in A.SYSTEM


def test_a_figure_the_maker_does_not_state_is_said_so_and_nothing_is_flagged(so100):
    ev, _, _ = turn(so100, [FakeReply(calls=[('get_sensor', {'sensor': WRIST})]),
                            FakeReply(text='The maker does not state that figure, so DexVantage shows it as not '
                                           'stated. I will not offer a typical value.')])
    a = answer(ev)
    assert a['check'] == {'checked': 0, 'unsupported': [], 'ok': True} and a['warning'] == ''


def test_an_invented_number_is_flagged_for_the_user(so100):
    ev, _, _ = turn(so100, [FakeReply(calls=[('get_sensor', {'sensor': WRIST})]),
                            FakeReply(text='Its minimum depth is 70 mm and it sees 87.5 % of the task volume.')])
    a = answer(ev)
    assert not a['check']['ok']
    assert {u['text'] for u in a['check']['unsupported']} >= {'70', '87.5'}
    assert '70' in a['warning'] and 'check them' in a['warning']


def test_a_figure_from_a_tool_or_the_page_is_not_flagged(so100):
    page = {'page': 'coverage', 'label': 'Coverage', 'headline': 'Shows 62.4 %: fair.',
            'shown': {'headline': {'figure': '62.4', 'unit': '%', 'word': 'fair'}}}

    def final(results):
        sensor = dict(results)['get_sensor']
        z, pitch = sensor['mount']['position_m'][2], sensor['mount']['rpy_deg'][1]
        return (f'The page shows 62.4 % (fair). The wrist camera sits at z = {z * 1000:.0f} mm, pitched '
                f'{pitch:.1f}°.')
    ev, _, _ = turn(so100, [FakeReply(calls=[('get_page_context', {}), ('get_sensor', {'sensor': WRIST})]),
                            FakeReply(text=final)], page=page)
    a = answer(ev)
    assert a['check']['ok'] and a['check']['checked'] == 3, a


def test_numcheck_reads_precision_restatements_and_skips_names():
    src = ['{"visible": 0.8754, "z": -0.0042, "angle": 0.5236, "count": 1886}']
    assert NC.unsupported('About 87.5 % are visible.', src)['ok']                 # 0.8754 as a percentage
    assert NC.unsupported('It sits 4.2 mm below.', src)['ok']                     # metres as mm, sign dropped
    assert NC.unsupported('Tilted 30°.', src)['ok']                               # radians as degrees
    assert NC.unsupported('1 886 frames.', src)['ok']                             # thousands with a space
    assert not NC.unsupported('2 000 frames.', src)['ok']                         # a number opening the answer
    assert NC.unsupported('Its uid is 3e512abc.', ['{"uid": "3e512abc"}'])['ok']   # an id, not 3 × 10^512
    assert not NC.unsupported('About 88.1 % are visible.', src)['ok']             # wrong in the last place
    assert not NC.unsupported('It sits 5 mm below.', src)['ok']                   # a small integer with a unit
    assert NC.unsupported('It sits 4 mm below.', src)['ok']                       # 0.0042 m, rounded
    assert NC.unsupported('Step 3 of 4 uses 2 cameras.', [])['checked'] == 0     # counts stay unchecked
    names = 'The UR5e and SO-100 on claude-opus-5-5 (v2.1, §7.3, 2026-11-07, `0.123`), step 3.'
    assert NC.unsupported(names, [])['checked'] == 0


# ======================================================================= edits are proposals
def test_propose_changes_is_a_proposal_with_a_diff_and_changes_nothing(so100):
    wrist = next(s for s in so100.variant.sensors if s.name == WRIST)
    z0 = wrist.mount.pose.position[2]
    ev, tc, _ = turn(so100, [FakeReply(calls=[('propose_changes', {'title': 'Raise the wrist camera',
                                                                   'changes': [{'factor': f'sensor:{wrist.uid}.z',
                                                                                'value': 0.05}]})]),
                             FakeReply(text='I proposed raising the wrist camera to 0.05 m.')])
    prop = next(e.data['proposal'] for e in ev if e.kind == 'proposal')
    assert prop.status == 'pending' and prop.rows[0]['before'] == pytest.approx(z0) and prop.rows[0]['after'] == 0.05
    assert tc.session.project().variant.sensors[0].mount.pose.position[2] == pytest.approx(z0)   # not applied
    p = so100.snapshot()
    keys = ED.apply(p, prop)
    assert keys == [f'variant:{p.active_uid}']
    assert next(s for s in p.variant.sensors if s.name == WRIST).mount.pose.position[2] == pytest.approx(0.05)


def test_a_changing_tool_comes_back_as_a_proposal_with_its_prediction(so100):
    ev, tc, _ = turn(so100, [FakeReply(calls=[('edit_project', {'changes': {f'sensor:{WRIST}.dz': 0.01}})]),
                             FakeReply(text='Proposed.')])
    res = json.loads(next(e.data['result'] for e in ev if e.kind == 'tool_done'))
    assert res['applied'] is False and res['proposals'] and 'predicted' in res
    assert len(tc.proposals) == 1 and tc.proposals[0].source == 'tool'
    before = next(s for s in so100.variant.sensors if s.name == WRIST).mount.pose.position[2]
    now = next(s for s in tc.session.project().variant.sensors if s.name == WRIST).mount.pose.position[2]
    assert now == pytest.approx(before)                                        # the project is unchanged


def test_a_tool_proposal_is_refused_after_the_project_changed(so100):
    _, tc, _ = turn(so100, [FakeReply(calls=[('edit_project', {'changes': {f'sensor:{WRIST}.dz': 0.01}})]),
                            FakeReply(text='Proposed.')])
    p = so100.snapshot()
    p.name = 'Renamed meanwhile'
    with pytest.raises(ValueError, match='changed after'):
        ED.apply(p, tc.proposals[0])


def test_a_locked_sensor_is_refused_in_plain_words(so100):
    p = so100.snapshot()
    p.variant.sensors[0].locked = True
    ev, _, _ = turn(p, [FakeReply(calls=[('propose_changes', {'title': 'Move it', 'changes': [
        {'factor': f'sensor:{p.variant.sensors[0].uid}.z', 'value': 0.3}]})]), FakeReply(text='It is locked.')])
    res = json.loads(next(e.data['result'] for e in ev if e.kind == 'tool_done'))
    assert 'locked' in res['error'].lower()


# ======================================================================= the Copilot gate
def test_a_long_study_is_refused_without_an_approved_plan(so100):
    assert CP.is_long('grasp_visibility')
    ev, _, _ = turn(so100, [FakeReply(calls=[('run_study', {'study': 'grasp_visibility'})]),
                            FakeReply(text='That needs a plan you approve.')])
    res = json.loads(next(e.data['result'] for e in ev if e.kind == 'tool_done'))
    assert res.get('plan_required') and 'plan' in res['error']
    ev, _, _ = turn(so100, [FakeReply(calls=[('export_files', {'formats': ['x']})]), FakeReply(text='ok')])
    assert 'plan' in json.loads(next(e.data['result'] for e in ev if e.kind == 'tool_done'))['error']


def test_plan_approval_gate(so100):
    ev, tc, _ = turn(so100, [FakeReply(calls=[('propose_plan', {'title': 'Check bandwidth', 'steps': [
        {'kind': 'study', 'study': 'bandwidth', 'settings': {'margin': 0.2}}]})]),
        FakeReply(text='The plan runs the bandwidth study.')], text='run the bandwidth study')
    plan = next(e.data['plan'] for e in ev if e.kind == 'plan')
    assert plan.status == 'draft' and plan.ok and plan.steps[0].settings == {'study': 'bandwidth',
                                                                             'settings': {'margin': 0.2}}
    step = plan.steps[0]
    p = so100.snapshot()
    # nothing is approved: the gate refuses
    ok, why = CP.gate(CP.records(p), {'plan': plan.id, 'version': 1, 'step': 0}, 'study', step.settings)
    assert not ok and 'No approval record' in why
    with pytest.raises(Exception, match='No approval record'):
        CP.run_step(p, CP.records(p), {'plan': plan.id, 'version': 1, 'step': 0})
    rec, errors = CP.approve(p, plan, 'fake', 'fake-model')
    assert errors == [] and CP.verify(rec)
    CP.add_record(p, rec)
    rs = CP.records(p)
    tk = CP.ticket(rec, 0)
    assert CP.gate(rs, tk, 'study', step.settings) == (True, '')
    assert not CP.gate(rs, tk, 'study', {'study': 'bandwidth', 'settings': {'margin': 0.5}})[0]   # other settings
    out = CP.run_step(p, rs, tk)
    assert out['study'] == 'bandwidth' and out['doc']['study'] == 'bandwidth'
    # a record changed after approval does not verify, and the gate refuses it
    bad = json.loads(json.dumps(rec))
    bad['steps'][0]['settings']['settings']['margin'] = 0.9
    assert not CP.verify(bad)
    assert 'does not verify' in CP.gate([bad], tk, 'study', bad['steps'][0]['settings'])[1]
    # an approved plan is never approved again; an edit is a new version with its diff
    plan.status = 'approved'
    assert CP.approve(p, plan)[0] is None
    v2 = CP.edit_value(p, plan, 0, 'margin', 0.3)
    assert v2.version == 2 and v2.status == 'draft'
    assert v2.diff == [{'step': 0, 'key': 'settings.margin', 'before': 0.2, 'after': 0.3}]


def test_a_plan_with_an_unknown_study_cannot_be_approved(so100):
    plan = CP.build(so100, [{'kind': 'study', 'study': 'no_such_study'}], 'Bad')
    assert not plan.ok and 'no study' in plan.errors[0]
    assert CP.approve(so100, plan)[0] is None


# ======================================================================= keys, chats, memory
def test_keys_live_in_the_keyring_only():
    SK.set_backend(None)
    assert os.environ.get('DEXVANTAGE_NO_KEYRING') and not SK.available()
    with pytest.raises(SK.KeyStoreError, match='keyring'):
        SK.save(PR.CLAUDE, 'test-key-123456789')
    assert SK.load(PR.CLAUDE) is None
    mem = SK.MemoryKeyring()
    SK.set_backend(mem)
    try:
        SK.save(PR.CLAUDE, ' test-key-123456789 ')
        assert SK.load(PR.CLAUDE) == 'test-key-123456789'
        assert mem.store == {('DexVantage', 'anthropic-api-key'): 'test-key-123456789'}
        assert SK.mask('test-key-123456789') == 'test…6789'
        SK.delete(PR.CLAUDE)
        assert SK.load(PR.CLAUDE) is None
    finally:
        SK.set_backend(None)
    # nothing about keys is read from the environment
    assert 'environ' not in open(SK.__file__, encoding='utf-8').read().split('def load')[1].split('def save')[0]


def test_chats_and_memory_are_kept_per_project_under_xdg_data_home(so100):
    root = ST.data_root()
    assert root.startswith(os.environ['XDG_DATA_HOME'])
    a, b = so100.snapshot(), so100.snapshot()
    b.name = 'Another project'
    cid = CH.new_id()
    CH.save(a, cid, title='Where is the wrist camera?', items=[{'role': 'user', 'text': 'Where?'},
                                                               {'role': 'assistant', 'text': 'Here.'}])
    assert [r['id'] for r in CH.listing(a)] == [cid] and CH.listing(b) == []
    assert oct(os.stat(os.path.join(CH.chat_dir(a), f'{cid}.json')).st_mode & 0o777) == '0o600'
    assert '### You' in CH.markdown(CH.load(a, cid))
    assert CH.delete(a, cid) and CH.listing(a) == []
    e = MM.add(a, MM.new_entry('constraint', 'Keep the third-person camera on the tripod.'))
    assert [x['id'] for x in MM.entries(a)] == [e['id']] and MM.entries(b) == []
    assert 'tripod' in MM.prompt_block(a) and MM.prompt_block(b) == ''
    assert MM.remove(a, e['id']) and MM.entries(a) == []


def test_remember_is_an_offer_not_a_write(so100):
    p = so100.snapshot()
    p.name = 'Memory offer project'
    ev, tc, _ = turn(p, [FakeReply(calls=[('remember', {'kind': 'decision', 'text': 'Use the D405 on the wrist.'})]),
                         FakeReply(text='Offered.')])
    assert tc.notes and tc.notes[0]['source'] == 'assistant'
    assert MM.entries(p) == []


# ======================================================================= errors and providers
def test_errors_are_plain_sentences_and_a_failed_turn_leaves_nothing(so100):
    off = ApiError('network', 'Could not reach Anthropic\'s API: you may be offline. Check the connection.')
    ev, _, s = turn(so100, [FakeReply(error=off)])
    err = next(e.data for e in ev if e.kind == 'error')
    assert err['kind'] == 'network' and 'offline' in err['message']
    assert ev[-1].data['ok'] is False and s.messages == []                      # rolled back
    rate = ApiError('rate', 'Anthropic is rate-limiting this key; try again in 1 s.', 429, retry_after=0.01)
    ev, _, _ = turn(so100, [FakeReply(error=rate), FakeReply(error=rate)])
    msgs = [e.data.get('text', '') for e in ev if e.kind == 'status']
    assert any('Rate limited' in m for m in msgs)
    assert next(e.data for e in ev if e.kind == 'error')['kind'] == 'rate'
    ev, _, _ = turn(so100, [FakeReply(stop='refusal', refusal={'category': None, 'explanation': None})])
    assert 'declined' in next(e.data for e in ev if e.kind == 'error')['message']
    with pytest.raises(ApiError, match='no Groq API key'):
        A.new_session(PR.GROQ, None, A.Settings('openai/gpt-oss-120b'))


def test_claude_requests_follow_the_documented_shape():
    pytest.importorskip('anthropic')
    s = A.new_session(PR.CLAUDE, 'test-key-123456789', A.Settings(PR.CLAUDE_DEFAULT))
    kw = s.request_kwargs()
    assert kw['model'] == 'claude-opus-5-5' and kw['thinking']['type'] == 'adaptive'
    assert kw['output_config'] == {'effort': 'medium'} and kw['tool_choice'] == {'type': 'auto'}
    assert 'temperature' not in kw and 'budget_tokens' not in json.dumps(kw['thinking'])
    assert kw['fallbacks'] == 'default' and 'server-side-fallback-2026-07-01' in kw['betas']
    assert all(t['eager_input_streaming'] for t in kw['tools'])
    assert [t['name'] for t in kw['tools']] == sorted(t['name'] for t in kw['tools'])
    s.settings.model, s.settings.planning = 'claude-haiku-5-5', True
    kw = s.request_kwargs()
    assert 'fallbacks' not in kw and kw['output_config'] == {'effort': 'high'}
    assert set(PR.models(PR.CLAUDE)) == {'claude-opus-5-5', 'claude-sonnet-5-5', 'claude-haiku-5-5'}
    with pytest.raises(ApiError, match='no Claude API key'):
        A.new_session(PR.CLAUDE, '', A.Settings(PR.CLAUDE_DEFAULT))


def test_claude_errors_map_to_plain_sentences():
    anthropic = pytest.importorskip('anthropic')
    httpx = pytest.importorskip("httpx2")       # the HTTP library the anthropic 2.x SDK uses
    from dexvantage.assistant import claude as C
    req = httpx.Request('POST', 'https://api.anthropic.invalid/v1/messages')
    r429 = httpx.Response(429, request=req, headers={'retry-after': '7'})
    e = C.map_error(anthropic.RateLimitError('rate', response=r429, body=None))
    assert e.kind == 'rate' and '7 s' in e.message
    e = C.map_error(anthropic.AuthenticationError('auth', response=httpx.Response(401, request=req), body=None))
    assert e.kind == 'auth' and 'Assistant settings' in e.message
    e = C.map_error(anthropic.APIConnectionError(request=req))
    assert e.kind == 'network' and 'offline' in e.message


def test_routing_and_usage():
    keys = {PR.CLAUDE: 'k', PR.GROQ: 'k'}
    assert RT.route('run the grasp study on every grasp', keys=keys).model == 'claude-opus-5-5'
    assert RT.route('what does the headline mean', keys=keys).provider == PR.GROQ
    r = RT.route('optimise the wrist camera mount', keys={PR.GROQ: 'k'})
    assert r.provider == PR.GROQ and 'no Claude key' in r.fallback
    assert RT.route('hello', keys={}) is None
    u = PR.Usage(provider=PR.CLAUDE, model='claude-opus-5-5')
    u.add('claude-opus-5-5', 1_000_000, 100_000)
    assert u.cost() == pytest.approx(4.0 + 2.0)
    u.add('claude-haiku-5-5', 10, 10, cache_read=5)
    assert u.cost() is None and 'not estimated' in u.line()                      # a price not stated: no cost
    h = PR.Usage(provider=PR.CLAUDE, model='claude-haiku-5-5')
    h.add('claude-haiku-5-5', 100_000, 1_000_000)                               # up to 100K: $0.10 / $0.50
    assert h.cost() == pytest.approx(0.01 + 0.5)
    h.add('claude-haiku-5-5', 200_000, 0)                                       # a longer prompt: $0.50 in
    assert h.cost() == pytest.approx(0.01 + 0.5 + 0.1)
