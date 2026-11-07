"""The Assistant dock (Window ▸ Assistant, Ctrl+J): a chat with Claude or Groq about the open project.

What it shows, top to bottom: the model picker and the dock's commands (new chat, past chats, memory notes,
settings); a notice when the assistant cannot answer yet (no key, no SDK, no keyring) with what to do; the
conversation: your questions, the answers (Markdown), the tools each answer called, a warning under an answer whose
figures were not all found in what the tools returned (the number post-check), edit proposals (the diff, Accept /
Discard, then Undo), Copilot plans (the steps and their checks, Approve / Edit / Cancel, then each step's progress)
and memory notes the assistant offers (Keep / Discard); the input box and the usage line.

AssistantController holds the state and does the work (the turn on a worker thread, everything that touches the
project on the UI thread); AssistantPanel is the view. The tests drive the controller with the fake provider
(use_script) and read the panel's widgets.

Errors are sentences (no key, offline, rate limit, a refusal); a traceback only goes to the log. Ported from Vantage
ui/assistant.py (the controller, the cards, the plan runner), much reduced: no agent mode, no pictures, no slash
commands."""
from __future__ import annotations

import json
import threading
import time
import traceback

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QFrame, QGridLayout,
                             QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu, QPlainTextEdit,
                             QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

from ..assistant import agent as A
from ..assistant import chats as CH
from ..assistant import copilot as CP
from ..assistant import edits as ED
from ..assistant import memory as MM
from ..assistant import providers as PR
from ..assistant import secrets as SK
from ..assistant import toolbox as TB
from ..assistant.client import ApiError
from .widgets import StatusLine, button, hline, label, tool_button

EDIT_PREFIX = 'Assistant ·'         # an accepted proposal's undo step
PLAN_PREFIX = 'Copilot ·'           # an approved plan's edit step


def _fmt(v, unit=''):
    if v is None:
        return '—'
    if isinstance(v, bool):
        return 'on' if v else 'off'
    if isinstance(v, float):
        if unit == 'm':
            return f'{v:.4f} m'
        return f'{v:.4g}{(" " + unit) if unit and unit != "°" else unit}'
    if isinstance(v, (list, dict)):
        s = json.dumps(v, default=str)
        return s if len(s) <= 60 else s[:59] + '…'
    return f'{v}{(" " + unit) if unit and unit != "°" else unit}'


# =========================================================================== the controller
class AssistantController(QObject):
    event = pyqtSignal(object)          # agent.Event, queued from the worker thread
    card = pyqtSignal(str, object)      # ('plan_status', plan_id) and the like: the view refreshes a card
    state_changed = pyqtSignal()

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.session = None
        self.session_key = None             # (provider, model) of the session
        self.script = None                  # the fake provider's script (tests)
        self.cancel = threading.Event()
        self.thread = None
        self.proposals: dict = {}
        self.plans: dict = {}               # plan id -> its latest copilot.Plan
        self.plan_state: dict = {}          # plan id -> {'status', 'steps': [text], 'record'}
        self.plan = None                    # the plan on screen (a follow-up revises it)
        self.notes: dict = {}               # offered memory entries by id
        self.items: list = []               # the transcript (saved per project)
        self.chat_id = CH.new_id()
        self.started = time.time()
        self.last_seconds: dict = {}
        self._tc = None
        self._runner = None
        ctx.analysis.finished.connect(self._study_finished)
        ctx.analysis.failed.connect(self._study_failed)
        ctx.store.project_replaced.connect(self.new_chat)
        ctx.on_copilot_plan = self.external_plan        # a plan proposed over MCP (assistant/session.StoreSession)

    # ---------------------------------------------------------------- settings
    @property
    def settings(self):
        return self.ctx.settings

    @property
    def provider(self) -> str:
        if self.script is not None:
            return PR.FAKE
        p = str(self.settings.value('assistant/provider', PR.CLAUDE) or PR.CLAUDE)
        return p if p in PR.PROVIDERS else PR.CLAUDE

    @property
    def model(self) -> str:
        if self.script is not None:
            return 'fake-model'
        m = str(self.settings.value(f'assistant/model_{self.provider}', '') or '')
        return m or PR.default_model(self.provider)

    @property
    def show_reasoning(self) -> bool:
        return str(self.settings.value('assistant/show_reasoning', 'false')).lower() in ('true', '1')

    def set_model(self, provider: str, model: str):
        self.settings.setValue('assistant/provider', provider)
        self.settings.setValue(f'assistant/model_{provider}', model)
        self.state_changed.emit()

    def use_script(self, script):
        """Tests and the self-test: answer with the fake provider; script is a list of turns (fake.FakeReply lists),
        queued after any turns not used yet."""
        if self.script is None:
            self.session = None
            self.script = []
        self.script += [list(t) for t in script]
        self.state_changed.emit()

    def blocker(self) -> str:
        """Why the assistant cannot answer now ('' when it can), as a sentence that says what to do."""
        p = self.provider
        if p == PR.FAKE:
            return ''
        if p == PR.CLAUDE and not PR.claude_available():
            return PR.INSTALL[PR.CLAUDE]
        if SK.load(p) is None:
            why = SK.why_unavailable()
            return (f'There is no {PR.LABEL[p]} API key: add one in Assistant settings (it is kept in the system '
                    'keyring).' + (f' {why}' if why else ''))
        return ''

    # ---------------------------------------------------------------- a question
    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def _session(self):
        key = (self.provider, self.model)
        if self.session is not None and self.session_key == key:
            return self.session
        st = A.Settings(self.model, show_reasoning=self.show_reasoning)
        if self.provider == PR.FAKE:
            self.session = A.new_session(PR.FAKE, None, st, script=[])
        else:
            self.session = A.new_session(self.provider, SK.load(self.provider), st)
        self.session_key = key
        return self.session

    def _context(self, text: str) -> TB.ToolContext:
        p = self.ctx.store.project
        page = None
        try:
            from ..assistant import page_guide as PG
            page = PG.current_page_state(self.ctx.window)
        except Exception:                               # noqa: BLE001 — the page context is a help, not a must
            traceback.print_exc()
        sel = []
        obj = self.ctx.store.selected()
        if obj is not None and hasattr(obj, 'name'):
            sel.append(f'{self.ctx.store.sel[0]} "{obj.name}" ({self.ctx.store.sel[1]})')
        tc = TB.ToolContext(project_name=p.name, variant_name=p.variant.name, page_context=page, selection=sel,
                            memory=MM.prompt_block(p), plan=self.plan, provider=self.provider, model=self.model,
                            request=text, last_seconds=dict(self.last_seconds))
        tc.session = TB.dock_session(self.ctx, tc)
        return tc

    def ask(self, text: str) -> bool:
        text = (text or '').strip()
        if not text or self.busy():
            return False
        why = self.blocker()
        if why:
            self.event.emit(A.Event('error', {'kind': 'no_key', 'message': why}))
            return False
        try:
            s = self._session()
        except ApiError as e:
            self.event.emit(A.Event('error', {'kind': e.kind, 'message': e.message}))
            return False
        s.settings.planning = CP.is_plan_request(text) or self.plan is not None
        if self.script:                                 # the fake provider: hand over the turns queued so far
            s.script += self.script
            self.script.clear()
        self._tc = self._context(text)
        self.items.append({'role': 'user', 'text': A.mask(text)})
        self.event.emit(A.Event('user', {'text': text}))
        self.cancel = threading.Event()
        emit = self.event.emit
        self.thread = threading.Thread(target=self._work, args=(s, text, self._tc, emit, self.cancel), daemon=True)
        self.thread.start()
        self.state_changed.emit()
        return True

    @staticmethod
    def _work(s, text, tc, emit, cancel):
        try:
            s.run_turn(text, tc, emit, cancel)
        except RuntimeError:                            # the dock was deleted meanwhile
            pass

    def stop(self):
        self.cancel.set()

    def wait(self, timeout=10.0) -> bool:
        """Tests: process events until the turn (and any plan run) ends."""
        from PyQt6.QtCore import QCoreApplication
        t0 = time.time()
        while time.time() - t0 < timeout:
            QCoreApplication.processEvents()
            if not self.busy() and self._runner is None:
                QCoreApplication.processEvents()
                return True
            time.sleep(0.005)
        return False

    def on_event(self, ev):
        """The controller's part of an event (the view adds its cards)."""
        d = ev.data
        if ev.kind == 'proposal':
            self.proposals[d['proposal'].id] = d['proposal']
        elif ev.kind == 'plan':
            pl = d['plan']
            self.plans[pl.id] = pl
            self.plan = pl
        elif ev.kind == 'memory':
            self.notes[d['entry']['id']] = d['entry']
        elif ev.kind == 'tool_done':
            self.items.append({'role': 'tool', 'text': d['name'], 'ok': d['ok']})
        elif ev.kind == 'answer':
            self.items.append({'role': 'assistant', 'text': A.mask(d['text']), 'flag': d.get('warning', '')})
        elif ev.kind == 'error':
            self.items.append({'role': 'note', 'text': d.get('message', '')})
        elif ev.kind == 'done':
            self.save_chat()
            QTimer.singleShot(0, self.state_changed.emit)

    # ---------------------------------------------------------------- chats
    def save_chat(self):
        first = next((i['text'] for i in self.items if i.get('role') == 'user'), '')
        try:
            CH.save(self.ctx.store.project, self.chat_id, title=CH.title_of(first), items=self.items,
                    provider=self.provider, model=self.model, started=self.started)
        except OSError:
            traceback.print_exc()

    def new_chat(self, *_):
        self.stop()
        self.session, self.session_key = None, None
        self.items, self.chat_id, self.started = [], CH.new_id(), time.time()
        self.proposals, self.plans, self.plan, self.notes = {}, {}, None, {}
        self.event.emit(A.Event('cleared', {}))
        self.state_changed.emit()

    def chats(self) -> list:
        return CH.listing(self.ctx.store.project)

    def open_chat(self, cid) -> dict | None:
        """Show a past conversation; the next question continues it with a hand-over summary."""
        rec = CH.load(self.ctx.store.project, cid)
        if rec is None:
            return None
        self.new_chat()
        from ..assistant import routing as RT
        self.items = list(rec.get('items') or [])
        self.chat_id, self.started = rec['id'], float(rec.get('started') or time.time())
        handover = RT.handover_text(CH.markdown(rec), rec.get('provider') or self.provider)
        try:
            self._session().handover = handover
        except ApiError:
            pass
        self.event.emit(A.Event('history', {'items': self.items}))
        return rec

    def delete_chat(self, cid) -> bool:
        return CH.delete(self.ctx.store.project, cid)

    # ---------------------------------------------------------------- proposals
    def accept(self, pid) -> str:
        """Apply a proposal as one undo step; returns '' or why not."""
        prop = self.proposals.get(pid)
        if prop is None or prop.status not in ('pending', 'undone'):
            return 'This proposal is not open any more.'
        st = self.ctx.store
        try:
            keys = ED.keys_for(st.project, prop)
            st.edit(keys, lambda p: ED.apply(p, prop), f'{EDIT_PREFIX} {prop.title}')
        except Exception as e:                          # noqa: BLE001 — FactorError / EditError: a plain sentence
            if not isinstance(e, ValueError):
                traceback.print_exc()
                return f'The change could not be made ({type(e).__name__}).'
            return str(e)
        prop.status, prop.undo_index = 'applied', st.undo.index()
        self.card.emit('proposal', pid)
        if self.ctx.window is not None:
            self.ctx.window.msg(f'Applied: {prop.title} (Edit ▸ Undo takes it back)', 5000)
        return ''

    def undo(self, pid) -> str:
        prop = self.proposals.get(pid)
        st = self.ctx.store
        if prop is None or prop.status != 'applied':
            return 'This proposal is not applied.'
        u = st.undo
        if u.index() != prop.undo_index or not u.command(u.index() - 1).text().startswith(EDIT_PREFIX):
            return 'Other changes came after it: undo them first, or use Edit ▸ Undo.'
        u.undo()
        prop.status = 'undone'
        self.card.emit('proposal', pid)
        return ''

    def discard(self, pid):
        prop = self.proposals.get(pid)
        if prop is None or prop.status != 'pending':
            return
        prop.status = 'discarded'
        e = MM.rejected_entry(prop)
        if e is not None:
            MM.add(self.ctx.store.project, e)
        self.card.emit('proposal', pid)

    # ---------------------------------------------------------------- memory
    def keep_note(self, eid) -> bool:
        e = self.notes.pop(eid, None)
        if e is None:
            return False
        MM.add(self.ctx.store.project, e)
        self.card.emit('note', eid)
        return True

    def discard_note(self, eid):
        self.notes.pop(eid, None)
        self.card.emit('note', eid)

    def memory(self) -> list:
        return MM.entries(self.ctx.store.project)

    def add_note(self, kind, text):
        return MM.add(self.ctx.store.project, MM.new_entry(kind, text, 'user'))

    def delete_note(self, eid) -> bool:
        return MM.remove(self.ctx.store.project, eid)

    # ---------------------------------------------------------------- plans
    def external_plan(self, plan: dict):
        """A plan proposed over MCP (tools.propose_plan through the window's session): shown as a plan card."""
        try:
            raw = [dict(s) for s in plan.get('steps') or [] if isinstance(s, dict)]
            p = CP.build(self.ctx.store.project, raw, plan.get('title', ''), plan.get('rationale', ''),
                         provider='claude', model='MCP client', last_seconds=self.last_seconds)
        except Exception:                               # noqa: BLE001
            traceback.print_exc()
            return
        self.event.emit(A.Event('plan', {'plan': p}))

    def edit_plan(self, pid, step: int, key: str, value) -> str:
        pl = self.plans.get(pid)
        if pl is None:
            return 'There is no such plan.'
        try:
            new = CP.edit_value(self.ctx.store.project, pl, step, key, CP.parse_value(value), self.last_seconds)
        except IndexError:
            return f'The plan has no step {step + 1}.'
        pl.status = 'superseded'
        self.card.emit('plan', pid)
        self.event.emit(A.Event('plan', {'plan': new}))
        return ''

    def cancel_plan(self, pid):
        pl = self.plans.get(pid)
        if pl is None:
            return
        if self._runner is not None and self._runner.rec['id'] == pid:
            self._runner.cancel()
            return
        if pl.status == 'draft':
            pl.status = 'cancelled'
            if self.plan is pl:
                self.plan = None
        self.card.emit('plan', pid)

    def records(self) -> list:
        return CP.records(self.ctx.store.project)

    def approve_plan(self, pid) -> str:
        """Approve the plan on screen and start it; returns '' or why not."""
        pl = self.plans.get(pid)
        if pl is None:
            return 'There is no such plan.'
        if self._runner is not None:
            return 'A plan is running: wait for it to end or cancel it.'
        project = self.ctx.store.project
        rec, errors = CP.approve(project, pl, self.provider, self.model, self.last_seconds)
        if rec is None:
            self.plan_state[pid] = {'status': 'refused', 'steps': errors}
            self.card.emit('plan', pid)
            return ' '.join(errors)
        CP.add_record(project, rec)
        pl.status = 'approved'
        self.plan = None
        self._runner = PlanRunner(self, rec)
        self._runner.start()
        return ''

    def _study_finished(self, sid, doc):
        if isinstance(doc, dict) and isinstance(doc.get('seconds'), (int, float)) and not doc.get('cached'):
            self.last_seconds[sid] = float(doc['seconds'])
        if self._runner is not None:
            self._runner.study_done(sid, doc, None)

    def _study_failed(self, sid, message):
        if self._runner is not None:
            from .store import failure_text
            self._runner.study_done(sid, None, failure_text(message, 'The study')[0])

    def _runner_done(self):
        self._runner = None
        self.state_changed.emit()


class PlanRunner(QObject):
    """Runs an approved plan's steps in order, each through the gate first: a study through the window's analysis
    controller (its settings for this run only), an export on a worker thread into a new folder, an edit as one undo
    step. Progress goes to the plan card; the record keeps each step's outcome."""

    def __init__(self, ctl: AssistantController, rec: dict):
        super().__init__()
        self.ctl, self.rec, self.i = ctl, rec, -1
        self.run = None
        self.waiting = None
        self.cancelled = False
        self.pid = rec['id']

    def _state(self, status, text=None):
        stt = self.ctl.plan_state.setdefault(self.pid, {'status': status, 'steps': [''] * len(self.rec['steps'])})
        stt['status'] = status
        stt['record'] = self.rec
        if text is not None and 0 <= self.i < len(stt['steps']):
            stt['steps'][self.i] = text
        self.ctl.card.emit('plan', self.pid)

    def start(self):
        self.run = CP.start_run(self.rec, 'app')
        self._save()
        self._state('running')
        QTimer.singleShot(0, self._next)

    def _save(self):
        try:
            CP.update_record(self.ctl.ctx.store.project, self.rec)
        except OSError:
            traceback.print_exc()

    def cancel(self):
        self.cancelled = True
        if self.waiting:
            self.ctl.ctx.analysis.cancel(self.waiting)
        self._finish('cancelled')

    def _finish(self, status):
        if self.rec.get('status') in ('done', 'failed', 'cancelled'):
            return
        CP.finish_run(self.rec, status)
        self._save()
        self._state(status)
        self.ctl._runner_done()

    def _step_result(self, ok: bool, text: str):
        self.run['steps'][self.i].update(status='done' if ok else 'failed', message=text)
        self._state('running', ('Done: ' if ok else 'Stopped: ') + text)
        self._save()
        if ok:
            QTimer.singleShot(0, self._next)
        else:
            self._finish('failed')

    def _next(self):
        if self.cancelled:
            return
        self.i += 1
        if self.i >= len(self.rec['steps']):
            self._finish('done')
            return
        s = self.rec['steps'][self.i]
        tk = CP.ticket(self.rec, self.i)
        ok, why = CP.gate(self.ctl.records(), tk, s['kind'], s['settings'])
        if not ok:
            self._step_result(False, why)
            return
        self.run['steps'][self.i]['status'] = 'running'
        self._state('running', f'Running: {s["label"]}…')
        ctx = self.ctl.ctx
        if s['kind'] == 'study':
            self.waiting = s['settings']['study']
            ctx.analysis.run(self.waiting, dict(s['settings'].get('settings') or {}))
        elif s['kind'] == 'edit':
            try:
                ed = CP.edit_project_after(ctx.store.project, s['settings'])
                ctx.store.edit(list(ed.keys), lambda p: [p.restore(k, ed.project.section(k)) for k in ed.keys],
                               f'{PLAN_PREFIX} {s["label"]}')
            except Exception as e:                      # noqa: BLE001 — EditError: a plain sentence
                self._step_result(False, str(e) if isinstance(e, ValueError) else type(e).__name__)
                return
            self._step_result(True, f'{s["label"]} (one undo step)')
        else:
            self._export(s)

    def _export(self, s):
        from .. import workdir
        from ..core import exporters as EX
        ctx = self.ctl.ctx
        project = ctx.store.project.snapshot()
        folder = workdir.project_dir(project, workdir.get(ctx.settings), 'exports',
                                     f'plan-{self.rec["id"]}-v{self.rec["version"]}')
        box = {}

        def work():
            try:
                box['man'] = EX.export(project, folder, s['settings']['formats'], s['settings'].get('options') or {})
            except Exception as e:                      # noqa: BLE001
                box['error'] = str(e) if isinstance(e, ValueError) else f'the export stopped ({type(e).__name__})'
        t = threading.Thread(target=work, daemon=True)
        t.start()

        def poll():
            if t.is_alive():
                QTimer.singleShot(50, poll)
                return
            if 'error' in box:
                self._step_result(False, box['error'])
                return
            man = box['man']
            bad = [k for k, v in man['formats'].items() if v.get('status') != 'ok']
            self._step_result(not bad, f'{len(man["files"])} files in {workdir.display(folder)}' +
                              (f'; refused: {", ".join(bad)}' if bad else ''))
        QTimer.singleShot(50, poll)

    def study_done(self, sid, doc, error):
        if self.waiting != sid or self.cancelled:
            return
        self.waiting = None
        if error:
            self._step_result(False, error)
            return
        sec = doc.get('seconds') if isinstance(doc, dict) else None
        self._step_result(True, f'{self.rec["steps"][self.i]["label"]}' +
                          (f' in {sec:.1f} s' if isinstance(sec, (int, float)) else '') +
                          ' (its page shows the result)')


# =========================================================================== the view
def _frame(name='card') -> QFrame:
    f = QFrame()
    f.setObjectName(name)
    f.setFrameShape(QFrame.Shape.StyledPanel)
    return f


class ProposalCard(QFrame):
    def __init__(self, ctl: AssistantController, prop: ED.Proposal):
        super().__init__()
        self.ctl, self.prop = ctl, prop
        self.setFrameShape(QFrame.Shape.StyledPanel)
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 6, 8, 6)
        v.addWidget(label(f'Proposed change: {prop.title}', bold=True, wrap=True))
        if prop.reason:
            v.addWidget(label(prop.reason, muted=True, wrap=True))
        g = QGridLayout()
        g.setHorizontalSpacing(8)
        for r, row in enumerate(prop.rows[:24]):
            g.addWidget(label(row.get('title', row.get('factor', '')), wrap=True), r, 0)
            if row.get('before') is None and row.get('after') is None and row.get('path', '').startswith('change'):
                continue
            g.addWidget(label(_fmt(row.get('before'), row.get('unit', '')), mono_=True, muted=True), r, 1)
            g.addWidget(label('→', muted=True), r, 2)
            g.addWidget(label(_fmt(row.get('after'), row.get('unit', '')), mono_=True), r, 3)
        if len(prop.rows) > 24:
            g.addWidget(label(f'… and {len(prop.rows) - 24} more', muted=True), 25, 0)
        v.addLayout(g)
        self.status = label('', muted=True, wrap=True)
        v.addWidget(self.status)
        h = QHBoxLayout()
        self.b_accept = button('Accept', role='primary', tip='Make this change (one undo step)')
        self.b_discard = button('Discard', role='ghost', tip='Leave the project as it is (remembered as rejected)')
        self.b_undo = button('Undo', role='ghost', icon_name='undo', tip='Take the change back')
        for b in (self.b_accept, self.b_discard, self.b_undo):
            h.addWidget(b)
        h.addStretch(1)
        v.addLayout(h)
        self.b_accept.clicked.connect(self._accept)
        self.b_discard.clicked.connect(lambda: ctl.discard(prop.id))
        self.b_undo.clicked.connect(self._undo)
        self.refresh()

    def _accept(self):
        why = self.ctl.accept(self.prop.id)
        if why:
            self.status.setText(why)

    def _undo(self):
        why = self.ctl.undo(self.prop.id)
        if why:
            self.status.setText(why)

    def refresh(self):
        s = self.prop.status
        self.b_accept.setVisible(s in ('pending', 'undone'))
        self.b_accept.setText('Apply again' if s == 'undone' else 'Accept')
        self.b_discard.setVisible(s == 'pending')
        self.b_undo.setVisible(s == 'applied')
        self.status.setText({'pending': 'Nothing is changed until you accept it.',
                             'applied': 'Applied as one undo step (Edit ▸ Undo also takes it back).',
                             'discarded': 'Discarded: the assistant will not propose it again (a memory note).',
                             'undone': 'Undone: the project is as it was.'}.get(s, ''))


class PlanCard(QFrame):
    def __init__(self, ctl: AssistantController, plan: CP.Plan):
        super().__init__()
        self.ctl, self.plan = ctl, plan
        self.setFrameShape(QFrame.Shape.StyledPanel)
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 6, 8, 6)
        v.addWidget(label(f'Copilot plan {plan.id} v{plan.version}: {plan.title}', bold=True, wrap=True))
        if plan.rationale:
            v.addWidget(label(plan.rationale, muted=True, wrap=True))
        self.step_labels = []
        for i, s in enumerate(plan.steps):
            v.addWidget(label(f'{i + 1}. {CP.KIND_LABEL.get(s.kind, s.kind)}: {s.label}', wrap=True))
            sets = s.settings.get('settings') if s.kind == 'study' else {k: s.settings[k] for k in s.settings
                                                                         if k not in ('study',)}
            for k, val in (sets or {}).items():
                src = (s.sources.get(k) or {}).get('source', '')
                v.addWidget(label(f'    {k} = {_fmt(val)}' + (f'  ({CP.SOURCES.get(src, src)})' if src else ''),
                                  mono_=True, muted=True, wrap=True))
            for e in s.errors:
                v.addWidget(StatusLine('critical', e))
            for w in s.warnings:
                v.addWidget(StatusLine('warning', w))
            est = s.estimate or {}
            if est.get('last_run_s') is not None:
                v.addWidget(label(f'    last run here: {est["last_run_s"]:.1f} s (a measurement, not a prediction)',
                                  muted=True))
            sl = label('', muted=True, wrap=True)
            v.addWidget(sl)
            self.step_labels.append(sl)
        if plan.diff:
            ch = '; '.join(f'step {d["step"] + 1} {d["key"]}: {_fmt(d.get("before"))} → {_fmt(d.get("after"))}'
                           if 'key' in d else f'step {d["step"] + 1} {"added" if "added" in d else "removed"}'
                           for d in plan.diff[:6])
            v.addWidget(label(f'Changed from v{plan.parent}: {ch}', muted=True, wrap=True))
        self.status = label('', wrap=True)
        v.addWidget(self.status)
        h = QHBoxLayout()
        self.b_approve = button('Approve', role='primary', icon_name='play',
                                tip='Run the plan as shown (nothing runs before you approve)')
        self.b_edit = button('Edit…', role='ghost', tip='Change a value: a new version, approved again')
        self.b_cancel = button('Cancel', role='ghost')
        for b in (self.b_approve, self.b_edit, self.b_cancel):
            h.addWidget(b)
        h.addStretch(1)
        v.addLayout(h)
        self.b_approve.clicked.connect(self._approve)
        self.b_edit.clicked.connect(self._edit)
        self.b_cancel.clicked.connect(lambda: ctl.cancel_plan(plan.id))
        self.refresh()

    def _approve(self):
        why = self.ctl.approve_plan(self.plan.id)
        if why:
            self.status.setText(why)

    def _edit(self):
        dlg = PlanEditDialog(self.plan, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            step, key, value = dlg.values()
            why = self.ctl.edit_plan(self.plan.id, step, key, value)
            if why:
                self.status.setText(why)

    def refresh(self):
        st = self.ctl.plan_state.get(self.plan.id) or {}
        status = st.get('status') or self.plan.status
        draft = self.plan.status == 'draft' and status in ('draft', 'refused')
        self.b_approve.setVisible(draft)
        self.b_approve.setEnabled(self.plan.ok)
        self.b_edit.setVisible(draft)
        self.b_cancel.setVisible(draft or status == 'running')
        for sl, text in zip(self.step_labels, st.get('steps') or []):
            sl.setText(text or '')
        words = {'draft': ('Nothing runs until you approve it.' if self.plan.ok else
                           'This plan has errors: it cannot be approved. Ask the assistant to fix it, or edit it.'),
                 'refused': 'Not approved: ' + ' '.join(st.get('steps') or []),
                 'running': 'Approved and running.', 'done': 'Done: every step ran.',
                 'failed': 'Stopped: a step could not run (see above).', 'cancelled': 'Cancelled.',
                 'superseded': 'Replaced by a newer version below.', 'approved': 'Approved.'}
        self.status.setText(words.get(status, status))


class PlanEditDialog(QDialog):
    def __init__(self, plan: CP.Plan, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f'Edit plan {plan.id}')
        f = QFormLayout(self)
        self.step = QComboBox()
        for i, s in enumerate(plan.steps):
            self.step.addItem(f'{i + 1}. {s.label}', i)
        self.key = QLineEdit()
        self.key.setPlaceholderText('a setting, e.g. n_samples (export: formats)')
        self.value = QLineEdit()
        self.value.setPlaceholderText('JSON: 0.5, true, "text", [1, 2]')
        f.addRow('Step', self.step)
        f.addRow('Setting', self.key)
        f.addRow('Value', self.value)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    def values(self):
        return int(self.step.currentData() or 0), self.key.text().strip(), self.value.text().strip()


class NoteCard(QFrame):
    def __init__(self, ctl: AssistantController, entry: dict):
        super().__init__()
        self.ctl, self.entry = ctl, entry
        self.setFrameShape(QFrame.Shape.StyledPanel)
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 6, 8, 6)
        v.addWidget(label(f'Remember this? {MM.KIND_LABEL[entry["kind"]]}: {entry["text"]}', wrap=True))
        h = QHBoxLayout()
        self.b_keep = button('Keep', role='primary')
        self.b_discard = button('Discard', role='ghost')
        h.addWidget(self.b_keep)
        h.addWidget(self.b_discard)
        h.addStretch(1)
        v.addLayout(h)
        self.status = label('', muted=True)
        v.addWidget(self.status)
        self.b_keep.clicked.connect(lambda: ctl.keep_note(entry['id']))
        self.b_discard.clicked.connect(lambda: ctl.discard_note(entry['id']))

    def refresh(self):
        kept = any(e['id'] == self.entry['id'] for e in self.ctl.memory())
        self.b_keep.hide()
        self.b_discard.hide()
        self.status.setText('Kept in the memory notes.' if kept else 'Not kept.')


class AnswerView(QLabel):
    """An answer in Markdown, word-wrapped, selectable."""

    def __init__(self, text=''):
        super().__init__()
        self.setTextFormat(Qt.TextFormat.MarkdownText)
        self.setWordWrap(True)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.raw = ''
        self.set_markdown(text)

    def set_markdown(self, text):
        self.raw = text
        self.setText(text)

    def append(self, text):
        self.set_markdown(self.raw + text)


class AssistantPanel(QWidget):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.ctl = AssistantController(ctx)
        self.cards: dict = {}               # ('proposal' | 'plan' | 'note', id) -> card
        self.current = None                 # the answer being streamed
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        head = QWidget()
        head.setFixedHeight(28)
        h = QHBoxLayout(head)
        h.setContentsMargins(8, 0, 4, 0)
        h.setSpacing(4)
        h.addWidget(label('Assistant'))
        self.model_box = QComboBox()
        self.model_box.setToolTip('The model that answers (Claude needs the anthropic package and a key)')
        h.addWidget(self.model_box, 1)
        self.b_new = tool_button('plus', 'New chat')
        self.b_chats = tool_button('history', 'Past chats of this project')
        self.b_memory = tool_button('checklist', 'Memory notes: what the assistant remembers for this project')
        self.b_settings = tool_button('gear', 'Assistant settings: keys (system keyring) and options')
        for b in (self.b_new, self.b_chats, self.b_memory, self.b_settings):
            h.addWidget(b)
        v.addWidget(head)
        v.addWidget(hline())
        self.notice = StatusLine('warning', '')
        self.notice.setContentsMargins(8, 4, 8, 4)
        v.addWidget(self.notice)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.body = QWidget()
        self.col = QVBoxLayout(self.body)
        self.col.setContentsMargins(8, 8, 8, 8)
        self.col.setSpacing(6)
        self.col.addStretch(1)
        self.scroll.setWidget(self.body)
        v.addWidget(self.scroll, 1)
        self.status = label('', muted=True, wrap=True)
        self.status.setContentsMargins(8, 2, 8, 2)
        v.addWidget(self.status)
        v.addWidget(hline())
        self.input = QPlainTextEdit()
        self.input.setPlaceholderText('Ask about the project or the page you are on (Ctrl+Enter sends)')
        self.input.setMaximumHeight(84)
        self.input.installEventFilter(self)
        row = QHBoxLayout()
        row.setContentsMargins(8, 4, 8, 6)
        row.addWidget(self.input, 1)
        self.b_send = button('Send', role='primary', icon_name='send', tip='Send (Ctrl+Enter)')
        self.b_stop = button('Stop', role='ghost', icon_name='stop', tip='Stop the answer')
        row.addWidget(self.b_send, 0, Qt.AlignmentFlag.AlignBottom)
        row.addWidget(self.b_stop, 0, Qt.AlignmentFlag.AlignBottom)
        v.addLayout(row)
        self.usage = label('', muted=True, small=True)
        self.usage.setContentsMargins(8, 0, 8, 4)
        v.addWidget(self.usage)
        self.b_send.clicked.connect(self.send)
        self.b_stop.clicked.connect(self.ctl.stop)
        self.b_new.clicked.connect(self.ctl.new_chat)
        self.b_chats.clicked.connect(self.chats_menu)
        self.b_memory.clicked.connect(self.open_memory)
        self.b_settings.clicked.connect(self.open_settings)
        self.model_box.activated.connect(self._model_picked)
        self.ctl.event.connect(self._on_event)
        self.ctl.card.connect(self._refresh_card)
        self.ctl.state_changed.connect(self.refresh)
        self.refresh()

    # ---------------------------------------------------------------- state
    def refresh(self):
        box = self.model_box
        box.blockSignals(True)
        box.clear()
        if self.ctl.provider == PR.FAKE:
            box.addItem('Fake model (tests)', (PR.FAKE, 'fake-model'))
        for p in PR.PROVIDERS:
            for m in PR.models(p):
                box.addItem(f'{PR.model_info(p, m).label} ({PR.LABEL[p]})', (p, m))
        cur = (self.ctl.provider, self.ctl.model)
        i = next((k for k in range(box.count()) if box.itemData(k) == cur), -1)
        if i < 0:
            box.addItem(f'{cur[1]} ({PR.LABEL.get(cur[0], cur[0])})', cur)
            i = box.count() - 1
        box.setCurrentIndex(i)
        box.blockSignals(False)
        why = self.ctl.blocker()
        self.notice.set('warning', why)
        self.notice.setVisible(bool(why))
        busy = self.ctl.busy()
        self.b_send.setEnabled(not busy)
        self.b_stop.setEnabled(busy)

    def _model_picked(self, i):
        d = self.model_box.itemData(i)
        if d and d[0] != PR.FAKE:
            self.ctl.script = None
            self.ctl.set_model(*d)

    def eventFilter(self, obj, ev):
        from PyQt6.QtCore import QEvent
        if obj is self.input and ev.type() == QEvent.Type.KeyPress and \
                ev.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and \
                ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.send()
            return True
        return super().eventFilter(obj, ev)

    def send(self, text=None):
        text = self.input.toPlainText() if text is None or isinstance(text, bool) else text
        if self.ctl.ask(text):
            self.input.clear()
        self.refresh()

    # ---------------------------------------------------------------- the transcript
    def _add(self, w):
        self.col.insertWidget(self.col.count() - 1, w)
        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()))
        return w

    def widgets(self) -> list:
        return [self.col.itemAt(i).widget() for i in range(self.col.count() - 1)]

    def _clear(self):
        for w in self.widgets():
            w.setParent(None)
            w.deleteLater()
        self.cards, self.current = {}, None

    def _on_event(self, ev):
        self.ctl.on_event(ev)
        d = ev.data
        k = ev.kind
        if k == 'cleared':
            self._clear()
            self.status.setText('')
            self.usage.setText('')
        elif k == 'history':
            for it in d['items']:
                self._item(it)
        elif k == 'user':
            self._add(self._bubble(d['text']))
            self.current = None
            self.status.setText('Thinking…')
        elif k == 'content':
            if self.current is None:
                self.current = self._add(AnswerView())
            self.current.append(d['text'])
        elif k == 'tool_start':
            self.status.setText(f'{d["label"]}…')
        elif k == 'tool_done':
            self.current = None
            self._add(label(('Tool: ' if d['ok'] else 'Tool refused: ') + TB.describe(d['name'])['label'] +
                            f' ({d["ms"]:.0f} ms)', muted=True, small=True))
        elif k == 'proposal':
            card = ProposalCard(self.ctl, d['proposal'])
            self.cards[('proposal', d['proposal'].id)] = self._add(card)
        elif k == 'plan':
            card = PlanCard(self.ctl, d['plan'])
            self.cards[('plan', d['plan'].id, d['plan'].version)] = self._add(card)
            self.cards[('plan', d['plan'].id)] = card
        elif k == 'memory':
            self.cards[('note', d['entry']['id'])] = self._add(NoteCard(self.ctl, d['entry']))
        elif k == 'answer':
            if self.current is None or self.current.raw.strip() != d['text'].strip():
                self.current = self._add(AnswerView(d['text']))
            if d.get('warning'):
                self._add(StatusLine('warning', d['warning']))
            self.current = None
        elif k == 'status':
            self.status.setText(d.get('text', ''))
        elif k == 'usage':
            u = d.get('usage')
            self.usage.setText(u.line() if u is not None else '')
        elif k == 'error':
            self._add(StatusLine('critical', d.get('message', 'The assistant could not answer.')))
            self.status.setText('')
        elif k == 'done':
            self.status.setText('')
            self.current = None
            self.refresh()

    def _bubble(self, text):
        w = label(text, wrap=True, ink2=True)
        w.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return w

    def _item(self, it):
        role = it.get('role')
        if role == 'user':
            self._add(self._bubble(it.get('text', '')))
        elif role == 'assistant':
            self._add(AnswerView(it.get('text', '')))
            if it.get('flag'):
                self._add(StatusLine('warning', it['flag']))
        elif role == 'note':
            self._add(StatusLine('critical', it.get('text', '')))

    def _refresh_card(self, kind, ident):
        if kind == 'plan':
            for key, card in self.cards.items():
                if key[0] == 'plan' and key[1] == ident and isinstance(card, PlanCard):
                    card.refresh()
            return
        card = self.cards.get((kind, ident))
        if card is not None:
            card.refresh()

    # ---------------------------------------------------------------- menus and dialogs
    def chats_menu(self):
        m = QMenu(self)
        rows = self.ctl.chats()
        if not rows:
            a = m.addAction('No past chats for this project')
            a.setEnabled(False)
        for r in rows[:30]:
            when = time.strftime('%Y-%m-%d %H:%M', time.localtime(float(r.get('updated') or 0)))
            m.addAction(f'{when}  {r.get("title") or "Conversation"}', lambda cid=r['id']: self.ctl.open_chat(cid))
        m.exec(self.b_chats.mapToGlobal(self.b_chats.rect().bottomLeft()))

    def open_memory(self):
        MemoryDialog(self.ctl, self).exec()

    def open_settings(self):
        if AssistantSettingsDialog(self.ctl, self).exec():
            self.ctl.session = None
        self.refresh()


class MemoryDialog(QDialog):
    """The memory notes for this project: see them, add one, delete any."""

    def __init__(self, ctl: AssistantController, parent=None):
        super().__init__(parent)
        self.ctl = ctl
        self.setWindowTitle('Memory notes')
        self.resize(520, 380)
        v = QVBoxLayout(self)
        v.addWidget(label('What the assistant is told about this project with every question. Delete what no longer '
                          'holds.', muted=True, wrap=True))
        self.list = QListWidget()
        v.addWidget(self.list, 1)
        h = QHBoxLayout()
        self.kind = QComboBox()
        for k in MM.KINDS:
            self.kind.addItem(MM.KIND_LABEL[k], k)
        self.text = QLineEdit()
        self.text.setPlaceholderText('A decision, constraint or note')
        b_add = button('Add')
        h.addWidget(self.kind)
        h.addWidget(self.text, 1)
        h.addWidget(b_add)
        v.addLayout(h)
        h2 = QHBoxLayout()
        self.b_delete = button('Delete', icon_name='trash')
        h2.addWidget(self.b_delete)
        h2.addStretch(1)
        b_close = button('Close', role='ghost')
        h2.addWidget(b_close)
        v.addLayout(h2)
        b_add.clicked.connect(self._add)
        self.b_delete.clicked.connect(self._delete)
        b_close.clicked.connect(self.accept)
        self.fill()

    def fill(self):
        self.list.clear()
        for e in self.ctl.memory():
            it = QListWidgetItem(f'{MM.KIND_LABEL[e["kind"]]} · {e["date"]} · {e["text"]}'
                                 + ('' if e['source'] == 'user' else '  (proposed by the assistant)'))
            it.setData(Qt.ItemDataRole.UserRole, e['id'])
            self.list.addItem(it)

    def _add(self):
        if self.ctl.add_note(self.kind.currentData(), self.text.text()):
            self.text.clear()
            self.fill()

    def _delete(self):
        it = self.list.currentItem()
        if it is not None:
            self.ctl.delete_note(it.data(Qt.ItemDataRole.UserRole))
            self.fill()


class AssistantSettingsDialog(QDialog):
    """The provider, the model and the keys. Keys go to the system keyring only and are shown masked."""

    def __init__(self, ctl: AssistantController, parent=None):
        super().__init__(parent)
        self.ctl = ctl
        self.setWindowTitle('Assistant settings')
        self.resize(560, 300)
        f = QFormLayout(self)
        self.provider = QComboBox()
        for p in PR.PROVIDERS:
            self.provider.addItem(PR.LABEL[p], p)
        self.model = QComboBox()
        f.addRow('Provider', self.provider)
        f.addRow('Model', self.model)
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_state = label('', muted=True, wrap=True)
        kr = QHBoxLayout()
        kr.addWidget(self.key, 1)
        self.b_save = button('Save key')
        self.b_remove = button('Remove key', role='ghost')
        kr.addWidget(self.b_save)
        kr.addWidget(self.b_remove)
        f.addRow('API key', kr)
        f.addRow('', self.key_state)
        self.reasoning = QCheckBox('Show a summary of the model\'s reasoning while it works')
        self.reasoning.setChecked(ctl.show_reasoning)
        f.addRow('', self.reasoning)
        f.addRow('', label('Questions, the page context and tool results are sent to the provider you pick; keys '
                           'and folder names are not.', muted=True, wrap=True))
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        f.addRow(bb)
        i = self.provider.findData(ctl.provider if ctl.provider in PR.PROVIDERS else PR.CLAUDE)
        self.provider.setCurrentIndex(max(0, i))
        self.provider.currentIndexChanged.connect(self._fill)
        self.b_save.clicked.connect(self._save)
        self.b_remove.clicked.connect(self._remove)
        self._fill()

    def _p(self):
        return self.provider.currentData()

    def _fill(self, *_):
        p = self._p()
        self.model.clear()
        for m in PR.models(p):
            info = PR.model_info(p, m)
            self.model.addItem(f'{info.label} · {info.note}' if info.note else info.label, m)
        cur = str(self.ctl.settings.value(f'assistant/model_{p}', '') or PR.default_model(p))
        self.model.setCurrentIndex(max(0, self.model.findData(cur)))
        self.key.setPlaceholderText(PR.KEY_HINT[p] + f' ({PR.KEYS_URL[p]})')
        self.key.clear()
        self._state()

    def _state(self, extra=''):
        p = self._p()
        k = SK.load(p)
        bits = [f'A key is stored in the system keyring ({SK.mask(k)}).' if k else 'No key is stored.']
        why = SK.why_unavailable()
        if why:
            bits.append(why)
        if p == PR.CLAUDE and not PR.claude_available():
            bits.append(PR.INSTALL[PR.CLAUDE])
        if extra:
            bits.insert(0, extra)
        self.key_state.setText(' '.join(bits))
        self.b_save.setEnabled(SK.available())
        self.b_remove.setEnabled(bool(k))

    def _save(self):
        try:
            SK.save(self._p(), self.key.text())
        except SK.KeyStoreError as e:
            self._state(str(e))
            return
        self.key.clear()
        self._state('Saved.')

    def _remove(self):
        SK.delete(self._p())
        self._state('Removed.')

    def _ok(self):
        self.ctl.settings.setValue('assistant/show_reasoning', 'true' if self.reasoning.isChecked() else 'false')
        self.ctl.script = None
        self.ctl.set_model(self._p(), self.model.currentData())
        self.accept()
