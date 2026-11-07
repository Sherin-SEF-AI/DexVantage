"""The Campaign page (Data ▸ Campaign): the data-collection campaign planner (core/campaign.py, the study 'campaign').

What it shows: the headline "N episodes over D days, S GB" with a status word against your budget (within budget,
tight above 90 % of a limit, over budget; "no budget set" without one), the settings (operators, demonstrations per
condition entered or from a power analysis, keep rate, times, the fatigue and day limits, the recording format, the
budget; each a field, one undo step per change, labelled "(default)" while it is DexVantage's), the conditions with
their recorded and expected kept episodes and storage, the design (a Williams balanced Latin square and which balance
holds), the budget figures and what to change, and the sessions day by day with the checklist of the one selected.
Apply writes the plan into the project (analysis.studies['campaign']['plan'], one undo step); Preview says what it
would write; Explain says how the schedule is built. Export writes the result as JSON with provenance; Export
files… writes the CSV, the iCalendar file and the printable HTML; Open spec… and Save spec… read and write the spec
file batch mode takes (--campaign SPEC.json), which also holds the levels the page does not edit (tasks, variants,
lighting presets, object sets, teleop devices, explicit conditions).

Known limits: those of core/campaign.py; the page has no 3D or Camera view overlay (a schedule has nothing to draw
in the scene).
"""
from __future__ import annotations

import copy
import json
import os
import traceback

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QComboBox, QFileDialog, QFormLayout, QHeaderView, QLineEdit, QMessageBox, QTreeWidget,
                             QTreeWidgetItem, QWidget)

from ...core import campaign as CP
from ..theme import STATUS
from ..widgets import button, label
from . import register
from .base import StudyPage, row

ROLE = Qt.ItemDataRole.UserRole
SHOWN_SESSIONS = 400                 # rows of the sessions list (filling it stays well under 100 ms)
# key path -> (label, kind, hint); kinds: list, int, num, prob, opt (number or blank), time, date, combo
FIELDS = (
    ('operators', 'Operators', 'list', 'names separated by commas'),
    ('demos.mode', 'Demonstrations from', 'combo', ''),
    ('demos.per_condition', 'Demonstrations per condition', 'int', 'kept episodes wanted per condition'),
    ('demos.p_base', 'Success rate now', 'prob', 'power analysis: the rate to compare against (0-1)'),
    ('demos.p_target', 'Success rate to detect', 'prob', 'power analysis: the rate a better condition reaches'),
    ('keep_rate', 'Keep rate', 'prob', 'share of episodes kept (your audit pass rate)'),
    ('episode_s', 'Episode length (s)', 'opt', 'blank: each task\'s trajectory duration'),
    ('reset_s', 'Reset between episodes (s)', 'num', ''),
    ('changeover_min', 'Changeover (min)', 'num', 'setting a condition up'),
    ('session_max_min', 'Session limit (min)', 'int', 'operator fatigue limit'),
    ('break_min', 'Break between sessions (min)', 'int', ''),
    ('sessions_per_day', 'Sessions per day', 'int', 'the rig'),
    ('operator_sessions_per_day', 'Sessions per operator per day', 'int', 'operator fatigue limit'),
    ('day_start', 'Day starts', 'time', 'HH:MM'),
    ('day_hours', 'Day length (h)', 'num', ''),
    ('start_date', 'First day', 'date', 'YYYY-MM-DD; blank: days numbered'),
    ('calibration.every', 'Calibration check', 'combo', ''),
    ('format', 'Recording format', 'combo', ''),
    ('compression_ratio', 'Compression ratio', 'opt', 'measured on a recorded episode (encoded / raw)'),
    ('budget.disk_gb', 'Budget: disk (GB)', 'opt', 'blank: not set'),
    ('budget.days', 'Budget: days', 'opt', 'blank: not set'),
    ('budget.upload_mbps', 'Budget: upload link (Mbit/s)', 'opt', 'blank: not set'),
    ('budget.upload_hours_per_day', 'Budget: upload hours per day', 'opt', 'blank: not set'),
    ('seed', 'Seed', 'int', 'shuffles the conditions and rows of the design'),
)
COMBOS = {'demos.mode': (('entered', 'Entered'), ('power', 'Power analysis')),
          'calibration.every': (('day', 'Start of each day'), ('session', 'Start of each session')),
          'format': tuple((f, CP.FORMAT_TITLE[f]) for f in CP.FORMATS)}


def _get(d, path):
    cur = d
    for k in path.split('.'):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _num(text):
    try:
        v = float(str(text).strip().replace(',', '.').replace('−', '-'))
    except ValueError:
        return None
    return v if v == v and abs(v) != float('inf') else None


class CampaignPage(StudyPage):
    """Campaign: which demonstrations to record, by whom and when. Every condition (task, rig variant, lighting,
    objects, teleop device) is recorded equally often by every operator, in blocks ordered by a Williams balanced
    Latin square (shuffled by the seed); sessions keep to the fatigue limit and days to their session and hour
    limits. Storage is the bandwidth study's data rate of the rig's sensors over the recorded time. The figure is the
    episodes, the days and the storage, judged against your budget."""
    STUDY = CP.STUDY_ID
    TITLE = 'Campaign'
    HEADLINE = 'Data-collection campaign'

    def __init__(self, ctx):
        self._building = False
        self.edits = {}
        super().__init__(ctx)

    def build_body(self, layout):
        self._building = True
        form_w = QWidget()
        form = QFormLayout(form_w)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(4)
        self.form_labels = {}
        for key, text, kind, hint in FIELDS:
            if kind == 'combo':
                w = QComboBox()
                for val, title in COMBOS[key]:
                    w.addItem(title, val)
                w.activated.connect(lambda _i, k=key, w=w: self.set_field(k, w.currentData()))
            else:
                w = QLineEdit()
                w.setPlaceholderText(hint)
                w.editingFinished.connect(lambda k=key, w=w: self.set_field(k, w.text()))
            w.setToolTip(hint)
            lab = label(text)
            self.form_labels[key] = lab
            self.edits[key] = w
            form.addRow(lab, w)
        layout.addWidget(label('Settings (each change is one undo step)', bold=True))
        layout.addWidget(form_w)
        self.b_open = button('Open spec…', role='ghost', tip='Read a campaign spec file (one undo step): it can hold '
                                                            'tasks, variants, lighting presets, object sets and '
                                                            'devices')
        self.b_save = button('Save spec…', role='ghost', tip='Write the spec as the file batch mode reads '
                                                            '(--campaign SPEC.json)')
        self.b_files = button('Export files…', role='ghost', icon_name='export',
                              tip='The plan as CSV, an iCalendar file and a printable HTML page')
        self.b_open.clicked.connect(lambda: self.open_spec())
        self.b_save.clicked.connect(lambda: self.save_spec())
        self.b_files.clicked.connect(lambda: self.export_files())
        layout.addWidget(row(self.b_open, self.b_save, self.b_files))
        layout.addWidget(label('Conditions', bold=True))
        self.conds = QTreeWidget()
        self.conds.setRootIsDecorated(False)
        self.conds.setHeaderLabels(['Id', 'Condition', 'Episode', 'Recorded', 'Expected kept', 'GB each'])
        self.conds.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.conds)
        self.design_label = label('', wrap=True)
        layout.addWidget(self.design_label)
        layout.addWidget(label('Budget', bold=True))
        self.budget = QTreeWidget()
        self.budget.setRootIsDecorated(False)
        self.budget.setHeaderLabels(['Figure', 'Plan', 'Limit', 'Share', 'Status'])
        self.budget.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.budget)
        self.advice = label('', wrap=True)
        layout.addWidget(self.advice)
        layout.addWidget(label('Sessions (click one for its checklist)', bold=True))
        self.sessions = QTreeWidget()
        self.sessions.setHeaderLabels(['Session', 'Day', 'Time', 'Operator', 'Episodes', 'GB'])
        self.sessions.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.sessions.itemClicked.connect(lambda it, _c: self.show_session(it.data(0, ROLE)))
        layout.addWidget(self.sessions)
        self.checklist = label('', wrap=True)
        self.checklist.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.checklist)
        self.notes = label('', muted=True, small=True, wrap=True)
        layout.addWidget(self.notes)
        self.ctx.store.project_replaced.connect(self._replaced)
        self.ctx.store.changed.connect(self._changed)
        self._building = False
        self.fill_form()

    # ---------------------------------------------------------------- the settings
    def stored(self) -> dict:
        return copy.deepcopy(self.ctx.store.project.analysis.studies.get(CP.STUDY_ID) or {})

    def fill_form(self):
        """The fields from the stored spec; a field left at its default shows the default and says so."""
        self._building = True
        try:
            p = self.ctx.store.project
            try:
                eff = CP.settings_of(p)
            except Exception:                           # noqa: BLE001 — a bad stored spec: show the raw values
                eff = copy.deepcopy(CP.DEFAULTS)
                eff['defaults'] = []
            given = self.stored()
            for key, text, kind, hint in FIELDS:
                w = self.edits[key]
                v = _get(eff, key)
                is_default = _get(given, key) is None
                if kind == 'combo':
                    i = w.findData(v)
                    w.setCurrentIndex(max(0, i))
                else:
                    if kind == 'list':
                        t = ', '.join(v or [])
                    elif key == 'compression_ratio':
                        t = '' if not v else f'{v["value"]:g}'
                    elif v is None:
                        t = ''
                    elif isinstance(v, float):
                        t = f'{v:g}'
                    else:
                        t = str(v)
                    w.setText(t)
                self.form_labels[key].setText(text + (' (default)' if is_default and kind != 'opt' and
                                                      key != 'start_date' else ''))
            # only the fields of the chosen source of demonstrations are editable (the others are not used)
            power = _get(eff, 'demos.mode') == 'power'
            for key, on in (('demos.per_condition', not power), ('demos.p_base', power), ('demos.p_target', power)):
                self.edits[key].setEnabled(on)
                self.form_labels[key].setEnabled(on)
        finally:
            self._building = False

    def set_field(self, key: str, value) -> bool:
        """Change one setting (one undo step) and run the plan again; a value that cannot be read is refused in
        words and nothing changes."""
        if self._building:
            return False
        kind = next(k for f, _t, k, _h in FIELDS if f == key)
        text = '' if value is None else str(value).strip()
        if kind == 'list':
            names = [x.strip() for x in text.split(',') if x.strip()]
            if not names:
                return self._refuse('Give at least one operator name.')
            new = names
        elif kind == 'combo':
            if value not in [v for v, _ in COMBOS[key]]:
                return False
            new = value
        elif kind in ('int', 'num', 'prob'):
            v = _num(text)
            if v is None:
                return self._refuse(f'"{text}" is not a number.')
            if kind == 'prob' and not 0 <= v <= 1:
                return self._refuse(f'{text} is not a share between 0 and 1.')
            if v < 0 or (kind == 'int' and key != 'seed' and v < 1):
                return self._refuse(f'{text} is too small here.')
            new = int(round(v)) if kind == 'int' else v
        elif kind == 'opt':
            if not text:
                new = None
            else:
                v = _num(text)
                if v is None or v <= 0:
                    return self._refuse(f'"{text}" is not a positive number (leave it blank for none).')
                new = {'value': v, 'source': 'your figure (Campaign page)'} if key == 'compression_ratio' else v
        elif kind == 'time':
            if CP.parse_hhmm(text) is None:
                return self._refuse(f'"{text}" is not a time of day such as 09:00.')
            new = text
        elif kind == 'date':
            if text and CP.parse_date(text) is None:
                return self._refuse(f'"{text}" is not a date such as 2026-11-09.')
            new = text or None
        else:
            return False
        cur = self.stored()
        if _get(cur, key) == new:
            return False
        parts = key.split('.')

        def fn(p):
            st = copy.deepcopy(p.analysis.studies.get(CP.STUDY_ID) or {})
            d = st
            for k in parts[:-1]:
                if not isinstance(d.get(k), dict):
                    d[k] = {}
                d = d[k]
            if new is None:
                d.pop(parts[-1], None)
            else:
                d[parts[-1]] = new
            p.analysis.studies[CP.STUDY_ID] = st
        title = next(t for f, t, _k, _h in FIELDS if f == key)
        self.ctx.store.edit('analysis', fn, f'Campaign · {title.lower()}')
        self.fill_form()
        self.run()
        return True

    def _refuse(self, text) -> bool:
        self.note.setText(text)
        self.fill_form()
        return False

    def _replaced(self):
        self.doc = None
        for w in (self.conds, self.budget, self.sessions):
            w.clear()
        self.checklist.setText('')
        self.suggestion_ready(False)
        self.fill_form()
        self.empty()

    def _changed(self, keys):
        if 'analysis' in keys and not self._building:
            self.fill_form()

    # ---------------------------------------------------------------- the result
    def show_result(self, doc):
        try:
            h = doc['headline']
            t = doc['totals']
            gb = f'{t["storage_gb"]:,.1f} GB' if not t.get('storage_bound') else f'≥ {t["storage_gb"]:,.1f} GB'
            self.head.set(f'{h["episodes"]:,}', f'episodes over {t["days"]} day{"s" if t["days"] != 1 else ""}, '
                          f'{gb}', h['status'], h['status_word'], h['explain'] + (
                              ' ' + ' '.join(doc['advice']) if doc.get('advice') else ''),
                          'Status against your budget: within budget, tight above 90 % of a limit (a default), over '
                          'budget; "no budget set" without one.')
            self.conds.clear()
            for c in doc['conditions']:
                it = QTreeWidgetItem([c['id'], c['label'], f'{c["episode_s"]:g} s', f'{c["recorded"]:,}',
                                      f'{c["expected_kept"]:,.1f}', f'{c["gb_per_episode"]:.3f}'
                                      + (' (at least)' if c['storage_incomplete'] else '')])
                it.setToolTip(2, c['episode_source'])
                tip = '; '.join(f'{CP.AXIS_TITLE[a]}: {c[a]}' for a in CP.AXES)
                it.setToolTip(1, tip)
                self.conds.addTopLevelItem(it)
            self.design_label.setText(doc['design']['text'])
            self.budget.clear()
            for c in doc['budget']['checks']:
                it = QTreeWidgetItem([c['what'], f'{c["value"]:,.2f} {c["unit"]}', f'{c["limit"]:g} {c["unit"]}',
                                      f'{c["share"] * 100:.0f} %', CP.WORDS[c['status']]])
                it.setForeground(4, QColor(STATUS[c['status']]))
                self.budget.addTopLevelItem(it)
            if not doc['budget']['checks']:
                self.budget.addTopLevelItem(QTreeWidgetItem(['No budget set', '—', '—', '—', '']))
            self.advice.setText(' '.join(doc.get('advice') or []))
            self.sessions.clear()
            for s in doc['sessions'][:SHOWN_SESSIONS]:
                it = QTreeWidgetItem([s['id'], f'{s["day"]}' + (f' · {s["date"]}' if s['date'] else ''),
                                      f'{s["start"]}–{s["end"]}', s['operator'], str(s['episodes']),
                                      f'{s["gb"]:.2f}'])
                it.setData(0, ROLE, s['index'])
                for b in s['blocks']:
                    ch = QTreeWidgetItem([b['condition'], '', f'{b["minutes"]:g} min', f'round {b["round"]}',
                                          str(b['episodes']), f'{b["gb"]:.2f}'])
                    ch.setToolTip(0, b['label'])
                    ch.setData(0, ROLE, s['index'])
                    it.addChild(ch)
                self.sessions.addTopLevelItem(it)
            if doc['sessions']:
                self.show_session(doc['sessions'][0]['index'])
            nt = list(doc.get('notes') or [])
            if len(doc['sessions']) > SHOWN_SESSIONS:
                nt.append(f'The first {SHOWN_SESSIONS} of {len(doc["sessions"]):,} sessions are listed; the exports '
                          'hold every one.')
            if doc.get('defaults'):
                nt.append('Left at their DexVantage default: ' + ', '.join(doc['defaults']) + '.')
            ap = CP.applied_plan(self.ctx.store.project)
            if ap is not None:
                same = ap['digest'] == CP.plan_digest(doc)
                nt.append('The project holds an applied plan' + (' (this one).' if same else
                                                                 ' that differs from this one.'))
            self.notes.setText(' '.join(nt))
        except (KeyError, TypeError, ValueError):
            traceback.print_exc()
        self.suggestion_ready(self.suggestion() is not None)

    def show_session(self, index):
        s = next((x for x in (self.doc or {}).get('sessions') or [] if x['index'] == index), None)
        if s is None:
            return
        self.checklist.setText(f'{s["id"]} · day {s["day"]} · {s["start"]}–{s["end"]} · {s["operator"]}\n'
                               + '\n'.join(f'☐ {c["text"]}' for c in s['checklist']))

    # ---------------------------------------------------------------- the suggestion: write the plan
    def suggestion(self):
        if not self.doc or self.doc.get('cancelled'):
            return None
        ap = CP.applied_plan(self.ctx.store.project)
        if ap is not None and ap.get('digest') == CP.plan_digest(self.doc):
            return None
        return self.doc

    def apply(self):
        doc = self.suggestion()
        if doc is None:
            return None
        self.ctx.store.edit('analysis', lambda p: CP.apply_plan(p, doc),
                            f'Campaign · write the plan ({doc["headline"]["text"]})')
        self.suggestion_ready(False)
        self.show_result(self.doc)
        return doc

    def preview(self):
        doc = self.suggestion()
        if self.ctx.window is not None:
            self.ctx.window.msg(f'Apply would write the plan into the project: {doc["headline"]["text"]}, '
                                f'{doc["totals"]["sessions"]} sessions (one undo step).' if doc else
                                'The project already holds this plan.', 8000)

    def explain(self):
        box = QMessageBox(self)
        box.setWindowTitle('How the campaign is planned')
        d = (self.doc or {}).get('design') or {}
        box.setText('Every condition is the combination of a task, a rig variant, a lighting preset, an object set '
                    'and a teleop device. Each operator records every condition equally often, in rounds; in a '
                    'round the conditions come in the order of one row of a Williams balanced Latin square, so each '
                    'condition is at each position, and after each other condition, equally often when the number '
                    'of sequences allows (the design line says which holds). The seed shuffles the conditions and '
                    'the rows. Episodes are packed into sessions within the fatigue limit, and sessions into days, '
                    'rotating which operator records first. Storage is the bandwidth study\'s data rate times the '
                    'recorded time. ' + (d.get('text') or ''))
        box.setModal(False)
        box.show()

    # ---------------------------------------------------------------- files
    def export_files(self, folder=None):
        """The CSV, the iCalendar file and the printable HTML into folder (asked for without one)."""
        if self.doc is None:
            if self.ctx.window is not None:
                self.ctx.window.msg('There is no campaign plan to export yet: Run it first.', 5000)
            return None
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the campaign plan')
            if not folder:
                return None
        try:
            paths = CP.write_files(self.doc, os.path.join(folder, 'campaign'))
        except CP.CampaignError as e:
            self.note.setText(str(e))
            return None
        if self.ctx.window is not None:
            self.ctx.window.msg(f'Exported the campaign plan: {", ".join(os.path.basename(p) for p in paths)}', 5000)
        return paths

    def open_spec(self, path=None) -> bool:
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, 'Open a campaign spec', '', 'JSON (*.json)')
            if not path:
                return False
        try:
            spec = CP.read_spec(path)
            # refuse what cannot be read or names what the project lacks, before changing anything
            CP.conditions(self.ctx.store.project, CP.settings_of(self.ctx.store.project, spec))
        except CP.CampaignError as e:
            self.note.setText(str(e))
            return False
        keep = (self.stored() or {}).get('plan')

        def fn(p):
            p.analysis.studies[CP.STUDY_ID] = {**copy.deepcopy(spec), **({'plan': keep} if keep else {})}
        self.ctx.store.edit('analysis', fn, f'Campaign · open the spec {os.path.basename(path)}')
        self.fill_form()
        self.run()
        return True

    def save_spec(self, path=None):
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, 'Save the campaign spec', 'campaign.json', 'JSON (*.json)')
            if not path:
                return None
        spec = {'schema': CP.SPEC_SCHEMA, **{k: v for k, v in self.stored().items() if k != 'plan'}}
        try:
            with open(path, 'w', encoding='utf-8') as fh:
                json.dump(spec, fh, indent=1, ensure_ascii=False)
                fh.write('\n')
        except OSError as e:
            self.note.setText(f'Could not write {os.path.basename(path)}: {e.strerror or e}.')
            return None
        return path


register('campaign', 'Campaign', CampaignPage, menu='Data', icon='timeline', action_id=None, workspace=None,
         order=88)
