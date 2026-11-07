"""The Assistant dock in the window, driven by the fake provider (no network, no key, no real keyring): it reads the
page you are on and cites it, flags numbers it cannot find, shows edits as proposals applied and undone as one undo
step, runs long work only from a plan you approve, keeps chats and memory per project, and explains errors."""
import json

import pytest

from dexvantage.assistant import chats as CH
from dexvantage.assistant import copilot as CP
from dexvantage.assistant import memory as MM
from dexvantage.assistant import secrets as SK
from dexvantage.assistant.client import ApiError
from dexvantage.assistant.fake import FakeReply
from dexvantage.core.robot.samples import sample_project
from dexvantage.ui.assistant import AnswerView, AssistantSettingsDialog, MemoryDialog, NoteCard, PlanCard, ProposalCard
from dexvantage.ui.widgets import StatusLine

from conftest import pump

WRIST = 'Wrist camera (sample placement)'


@pytest.fixture
def dock(app, win):
    win.store.set_project(sample_project('so100'))
    win.ctx.analysis.wait(30)
    win.docks['assistant'].show()
    pump(app, 0.05)
    return win.assistant


def ask(app, dock, text, script):
    dock.ctl.use_script([script])
    dock.input.setPlainText(text)
    dock.send()
    assert dock.ctl.wait(30)
    pump(app, 0.05)


def of(dock, cls):
    return [w for w in dock.widgets() if isinstance(w, cls)]


def warnings(dock):
    return [w.text.text() for w in of(dock, StatusLine) if 'Not found' in w.text.text()]


def test_the_dock_is_in_the_window_menu_and_says_when_it_has_no_key(app, win):
    a = win.actions_by_id['assistant']
    assert a.text() == 'Assistant' and 'Ctrl+J' in a.shortcut().toString()
    d = win.docks['assistant']
    a.trigger()
    pump(app)
    assert d.isVisible()
    dock = win.assistant
    assert dock.ctl.provider == 'claude' and dock.ctl.model == 'claude-opus-5-5'
    assert {dock.model_box.itemData(i)[1] for i in range(dock.model_box.count())} >= {
        'claude-opus-5-5', 'claude-sonnet-5-5', 'claude-haiku-5-5'}
    why = dock.ctl.blocker()
    assert why and ('API key' in why or 'anthropic' in why) and dock.notice.isVisibleTo(dock)
    assert not dock.ctl.ask('hello')                                        # explained, nothing sent
    pump(app)
    assert of(dock, StatusLine) and 'Traceback' not in of(dock, StatusLine)[-1].text.text()


def test_it_reads_the_page_on_screen_and_cites_it(app, win, dock):
    win.open_page('coverage')
    win.pages['coverage'].run()                                             # about 1 s on the SO-100
    win.ctx.analysis.wait(60)
    pump(app, 0.1)
    shown = win.pages['coverage'].page_context()['headline']
    assert shown.get('figure') not in (None, '', '—'), shown

    def final(results):
        h = dict(results)['get_page_context']['shown']['headline']
        return f'The page shows {h["figure"]} {h.get("unit", "")} ({h.get("word", "")}).'
    ask(app, dock, 'What does this page say?', [FakeReply(calls=[('get_page_context', {})]), FakeReply(text=final)])
    answers = of(dock, AnswerView)
    assert answers and shown['figure'] in answers[-1].raw
    assert warnings(dock) == []                                             # the figure came from the page
    tc = dock.ctl._tc
    assert tc.page_context['page'] == 'coverage' and tc.page_context['status'] == 'done'
    assert shown['figure'] in json.dumps(tc.page_context)


def test_an_invented_number_is_shown_to_the_user(app, dock):
    ask(app, dock, 'How far can the wrist camera see?',
        [FakeReply(calls=[('get_sensor', {'sensor': WRIST})]), FakeReply(text='It sees up to 1.85 m.')])
    w = warnings(dock)
    assert w and '1.85' in w[0]


def test_an_edit_proposal_is_applied_and_undone_as_one_step(app, win, dock):
    st = win.store
    wrist = next(s for s in st.project.variant.sensors if s.name == WRIST)
    z0, n0 = float(wrist.mount.pose.position[2]), st.undo.count()
    ask(app, dock, 'Raise the wrist camera to 5 cm', [
        FakeReply(calls=[('propose_changes', {'title': 'Raise the wrist camera', 'changes': [
            {'factor': f'sensor:{wrist.uid}.z', 'value': 0.05}]})]),
        FakeReply(text='I proposed z = 0.05 m for the wrist camera.')])
    card = of(dock, ProposalCard)[-1]
    assert card.b_accept.isVisibleTo(card) and st.undo.count() == n0           # nothing changed yet

    def z():
        return float(next(s for s in st.project.variant.sensors if s.name == WRIST).mount.pose.position[2])
    assert z() == pytest.approx(z0)
    card.b_accept.click()
    pump(app)
    assert z() == pytest.approx(0.05) and st.undo.count() == n0 + 1
    assert st.undo.command(st.undo.index() - 1).text() == 'Assistant · Raise the wrist camera'
    assert card.b_undo.isVisibleTo(card) and not card.b_accept.isVisibleTo(card)
    card.b_undo.click()
    pump(app)
    assert z() == pytest.approx(z0) and card.prop.status == 'undone'
    card.b_accept.click()                                                       # "Apply again", then Edit ▸ Undo
    st.undo.undo()
    assert z() == pytest.approx(z0)


def test_a_changing_tool_never_edits_directly(app, win, dock):
    st = win.store
    n0 = st.undo.count()
    ask(app, dock, 'Move it down 1 cm', [
        FakeReply(calls=[('edit_project', {'changes': {f'sensor:{WRIST}.dz': -0.01}})]), FakeReply(text='Proposed.')])
    assert st.undo.count() == n0 and of(dock, ProposalCard)
    assert of(dock, ProposalCard)[-1].prop.source == 'tool'


def test_a_discarded_proposal_is_remembered_and_sent_next_time(app, win, dock):
    wrist = next(s for s in win.store.project.variant.sensors if s.name == WRIST)
    ask(app, dock, 'Tilt it', [FakeReply(calls=[('propose_changes', {'title': 'Tilt the wrist camera', 'changes': [
        {'factor': f'sensor:{wrist.uid}.pitch', 'value': -40.0}]})]), FakeReply(text='Proposed.')])
    card = of(dock, ProposalCard)[-1]
    card.b_discard.click()
    notes = MM.entries(win.store.project)
    assert notes and notes[-1]['kind'] == 'rejected' and 'Tilt the wrist camera' in notes[-1]['text']
    ask(app, dock, 'Anything else?', [FakeReply(text='No.')])
    assert 'Tilt the wrist camera' in dock.ctl._tc.memory


def test_long_work_runs_only_from_an_approved_plan(app, win, dock):
    ask(app, dock, 'Run the grasp study', [FakeReply(calls=[('run_study', {'study': 'grasp_visibility'})]),
                                           FakeReply(text='It needs a plan.')])
    assert 'grasp_visibility' not in win.ctx.analysis.results
    ask(app, dock, 'Plan the bandwidth study then', [
        FakeReply(calls=[('propose_plan', {'title': 'Bandwidth check', 'steps': [
            {'kind': 'study', 'study': 'bandwidth', 'settings': {'margin': 0.25}}]})]),
        FakeReply(text='The plan runs the bandwidth study once.')])
    card = of(dock, PlanCard)[-1]
    plan = card.plan
    assert card.b_approve.isVisibleTo(card) and plan.status == 'draft'
    assert CP.records(win.store.project) == []                                   # nothing approved, nothing ran
    # Edit: a new version with its diff; the old one is superseded
    assert dock.ctl.edit_plan(plan.id, 0, 'margin', '0.3') == ''
    pump(app)
    v2 = of(dock, PlanCard)[-1]
    assert v2.plan.version == 2 and plan.status == 'superseded' and not card.b_approve.isVisibleTo(card)
    assert v2.plan.diff[0]['after'] == 0.3
    v2.b_approve.click()
    assert dock.ctl.wait(60)
    pump(app, 0.1)
    rec = CP.find(CP.records(win.store.project), plan.id, 2)
    assert rec is not None and rec['status'] == 'done' and CP.verify(rec)
    assert rec['runs'][-1]['steps'][0]['status'] == 'done'
    assert 'Done' in v2.status.text()
    assert CP.approve(win.store.project, v2.plan)[0] is None                      # approved once only


def test_a_plan_can_be_cancelled_before_it_runs(app, dock):
    ask(app, dock, 'Plan an export', [FakeReply(calls=[('propose_plan', {'title': 'Export', 'steps': [
        {'kind': 'study', 'study': 'bandwidth'}]})]), FakeReply(text='Planned.')])
    card = of(dock, PlanCard)[-1]
    card.b_cancel.click()
    pump(app)
    assert card.plan.status == 'cancelled' and not card.b_approve.isVisibleTo(card)


def test_chats_are_saved_per_project_and_reopened(app, win, dock):
    ask(app, dock, 'First question', [FakeReply(text='First answer.')])
    rows = dock.ctl.chats()
    assert rows and rows[0]['title'] == 'First question'
    dock.ctl.new_chat()
    pump(app)
    assert dock.widgets() == []
    dock.ctl.open_chat(rows[0]['id'])
    pump(app)
    assert any(isinstance(w, AnswerView) and 'First answer' in w.raw for w in dock.widgets())
    assert dock.ctl.session.handover.startswith('[Earlier in this conversation')
    assert CH.listing(win.store.project)[0]['id'] == rows[0]['id']


def test_memory_notes_offered_kept_viewed_and_deleted(app, win, dock):
    ask(app, dock, 'We always use the D405 on the wrist', [
        FakeReply(calls=[('remember', {'kind': 'decision', 'text': 'Use the D405 on the wrist.'})]),
        FakeReply(text='Offered as a memory note.')])
    card = of(dock, NoteCard)[-1]
    assert MM.entries(win.store.project) == [] or all('D405' not in e['text'] for e in MM.entries(win.store.project))
    card.b_keep.click()
    pump(app)
    assert any('D405' in e['text'] for e in MM.entries(win.store.project))
    dlg = MemoryDialog(dock.ctl, dock)
    assert dlg.list.count() == len(MM.entries(win.store.project))
    dlg.list.setCurrentRow(0)
    dlg.b_delete.click()
    assert dlg.list.count() == len(MM.entries(win.store.project))
    dlg.deleteLater()


def test_keys_are_entered_in_the_settings_dialog_into_the_keyring(app, win):
    mem = SK.MemoryKeyring()
    SK.set_backend(mem)
    try:
        dock = win.assistant
        dlg = AssistantSettingsDialog(dock.ctl, dock)
        assert dlg.key.echoMode().name == 'Password' and dlg.b_save.isEnabled()
        dlg.key.setText('test-key-123456789')
        dlg.b_save.click()
        assert SK.load('claude') == 'test-key-123456789' and 'test…6789' in dlg.key_state.text()
        assert dlg.key.text() == ''
        dlg.model.setCurrentIndex(dlg.model.findData('claude-haiku-5-5'))
        dlg._ok()
        assert dock.ctl.model == 'claude-haiku-5-5'
        dlg.b_remove.click()
        assert SK.load('claude') is None
        dlg.deleteLater()
    finally:
        SK.set_backend(None)


def test_errors_are_explained_in_the_dock(app, dock):
    ask(app, dock, 'Hello?', [FakeReply(error=ApiError('network', 'Could not reach Anthropic\'s API: you may be '
                                                                  'offline. Check the connection and try again.'))])
    lines = [w.text.text() for w in of(dock, StatusLine)]
    assert any('offline' in t for t in lines) and not any('Traceback' in t for t in lines)
    assert dock.b_send.isEnabled()
