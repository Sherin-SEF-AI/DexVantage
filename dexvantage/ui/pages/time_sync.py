"""The Time sync page (spec §7.10): how far each camera's frame stamps are from the instant the frame was exposed,
against the robot's joint_states, and the action-label error that skew gives (core/studies/time_sync.py).

What it shows: the worst skew of a camera stream against joint_states (the headline, with a status word: "in sync"
within the skew limit, 2 ms by default; "tight" up to twice it; "out of sync" beyond; "incomplete" or "at least" when
a term of the budget is not stated), a table of the streams (clock, trigger, whether the maker states a hardware
trigger, the skew interval, its spread and the action-label error), the budget of the selected stream term by term
with where each figure comes from and what is not stated, joint_states' own budget, the camera pairs, and the settings:
the limit, the episode length, how labels are paired, joint_states' rate, each stream's clock and trigger, and the
rig's clocks with their figures and sources (project.rig['sync']). Apply makes the suggested trigger or sync cable
(one undo step); Preview says what it would change; Show worst moment poses the robot where the label error is
largest. The 3D view draws the TCP path and, at the worst moment, the TCP at the exposure and at the stamp joined; the
Camera view draws that segment too.

Known limits: see the study's (device processing times are not stated, so host-stamped budgets are lower bounds).
"""
from __future__ import annotations

import math
import traceback

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QHeaderView, QLineEdit, QMessageBox, QTreeWidget,
                             QTreeWidgetItem, QWidget)

from ...core.project import CLOCK_KINDS, TRIGGERS, new_uid
from ...core.studies import time_sync as TS
from ..theme import STATUS, THEME, mono
from ..widgets import ScrubSpinBox, button, label
from . import register
from .base import StudyPage, row

SEV_COLOUR = {'good': 'good', 'warning': 'warning', 'serious': 'serious', 'unknown': 'warning'}
TRIGGER_TEXT = {'free': 'free-running', 'hardware': 'hardware trigger', 'sync_cable': 'sync cable'}
KIND_TEXT = {'camera_hw': 'camera hardware clock', 'host': 'host clock', 'ptp': 'PTP', 'ntp': 'NTP',
             'robot': 'robot controller clock'}


def _ms(x, d=2) -> str:
    return '—' if x is None else f'{x * 1000:.{d}f}'


def _mm(x) -> str:
    return '—' if x is None else f'{x * 1000:.2f}'


def _yes(x) -> str:
    return '—' if x is None else ('yes' if x else 'no')


def _num(text: str):
    """A typed number, or None for blank or not a finite number."""
    try:
        v = float(str(text).replace(',', '.').replace('−', '-'))
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


class TimeSyncPage(StudyPage):
    """Time sync: each camera frame's stamp error (stamp minus the instant of exposure) is summed from the exposure,
    the readout, the USB transfer behind the other devices on its controller (the bandwidth topology), the 125 µs bus
    interval and the clocks' stated figures; the skew against joint_states, times the end effector's motion along the
    trajectory, is the action-label error. The headline is the worst skew, against the skew limit (2 ms by
    default)."""
    STUDY = 'time_sync'
    TITLE = 'Time sync'
    HEADLINE = 'Worst skew of a camera against joint_states'

    # ---------------------------------------------------------------- body
    def build_body(self, layout):
        self._overlay3d = None
        layout.addWidget(label('Streams', bold=True))
        self.streams = QTreeWidget()
        self.streams.setRootIsDecorated(False)
        self.streams.setHeaderLabels(['Camera', 'Clock', 'Trigger', 'Hardware trigger stated', 'Skew (ms)',
                                      'Spread (ms)', 'Label error (mm)', 'Status'])
        _columns(self.streams)
        self.streams.currentItemChanged.connect(lambda *_a: self._fill_budget())
        layout.addWidget(self.streams)
        layout.addWidget(label('Budget of the selected stream', bold=True))
        self.budget = QTreeWidget()
        self.budget.setRootIsDecorated(False)
        self.budget.setHeaderLabels(['Term', 'Low (ms)', 'High (ms)', 'From'])
        _columns(self.budget)
        self.budget.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.budget.header().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.budget)
        self.missing = label('', wrap=True)
        layout.addWidget(self.missing)
        self.joint_label = label('', muted=True, wrap=True)
        layout.addWidget(self.joint_label)
        layout.addWidget(label('Camera pairs', bold=True))
        self.pairs = QTreeWidget()
        self.pairs.setRootIsDecorated(False)
        self.pairs.setHeaderLabels(['Cameras', 'Phase (ms)', 'Worst offset (ms)', 'Locked', 'Sync cable possible'])
        _columns(self.pairs)
        layout.addWidget(self.pairs)
        self.b_worst = button('Show worst moment', role='ghost', icon_name='eye',
                              tip='Pose the robot where the action-label error of the selected stream is largest')
        self.b_worst.clicked.connect(lambda: self.show_worst())
        self.show3d = QCheckBox('Show the TCP path and the label error in 3D')
        self.show3d.setChecked(True)
        self.show3d.toggled.connect(lambda _on: self._redraw())
        layout.addWidget(row(self.b_worst, self.show3d))
        self.sugg_label = label('', muted=True, wrap=True)
        layout.addWidget(self.sugg_label)
        self.notes_label = label('', muted=True, wrap=True)
        layout.addWidget(self.notes_label)
        # ---- settings
        layout.addWidget(label('Settings (yours; defaults labelled)', bold=True))
        g = QWidget()
        grid = QGridLayout(g)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        self.limit_spin = ScrubSpinBox('', 'plain', 2, 'ms', step=0.1, width=110)
        self.limit_spin.setRange(0.001, 10000.0)
        self.limit_spin.setToolTip(f'The most skew accepted (default {TS.DEFAULTS["skew_limit_s"] * 1000:g} ms, a '
                                   'DexVantage default, not a standard)')
        self.limit_def = label('', muted=True, small=True)
        self.limit_spin.committed.connect(lambda v: self._set_setting('skew_limit_s', float(v) / 1000.0,
                                                                      f'Time sync · skew limit {float(v):g} ms'))
        self.ep_spin = ScrubSpinBox('', 'plain', 1, 's', step=1.0, width=110)
        self.ep_spin.setRange(0.1, 86400.0)
        self.ep_spin.setToolTip(f'One episode: the time a drifting clock walks over (default '
                                f'{TS.DEFAULTS["episode_s"]:g} s)')
        self.ep_def = label('', muted=True, small=True)
        self.ep_spin.committed.connect(lambda v: self._set_setting('episode_s', float(v),
                                                                   f'Time sync · episode {float(v):g} s'))
        self.pair_box = QComboBox()
        self.pair_box.addItem('Interpolate joint_states (default)', 'interpolate')
        self.pair_box.addItem('Nearest joint_states sample', 'nearest')
        self.pair_box.activated.connect(lambda _i: self._set_setting('pairing', self.pair_box.currentData(),
                                                                     'Time sync · label pairing'))
        self.rate_edit = QLineEdit()
        self.rate_edit.setPlaceholderText('Hz')
        self.rate_edit.setMaximumWidth(90)
        self.rate_src = QLineEdit()
        self.rate_src.setPlaceholderText('where the rate comes from')
        self.b_rate = button('Set', role='ghost', tip='The joint_states rate and its source (blank: not stated)')
        self.b_rate.clicked.connect(lambda: self.set_joint_rate(_num(self.rate_edit.text()), self.rate_src.text()))
        for r, (name, w, d) in enumerate((('Skew limit', self.limit_spin, self.limit_def),
                                          ('Episode', self.ep_spin, self.ep_def))):
            grid.addWidget(label(name), r, 0)
            grid.addWidget(w, r, 1)
            grid.addWidget(d, r, 2)
        grid.addWidget(label('Labels'), 2, 0)
        grid.addWidget(self.pair_box, 2, 1, 1, 2)
        grid.addWidget(label('joint_states rate'), 3, 0)
        grid.addWidget(row(self.rate_edit, self.rate_src, self.b_rate), 3, 1, 1, 2)
        layout.addWidget(g)
        # ---- each stream's clock and trigger
        layout.addWidget(label('Clock and trigger of a stream', bold=True))
        self.s_box = QComboBox()
        self.s_box.setToolTip('A sensor, or joint_states')
        self.c_box = QComboBox()
        self.c_box.setToolTip('The clock its stamps are taken on; host arrival is the default')
        self.t_box = QComboBox()
        for k in TRIGGERS:
            self.t_box.addItem(TRIGGER_TEXT[k], k)
        self.b_stream = button('Set', role='ghost', tip='Set this stream\'s clock and trigger (one undo step)')
        self.b_stream.clicked.connect(lambda: self.set_stream(self.s_box.currentData(), self.c_box.currentData(),
                                                              self.t_box.currentData()))
        layout.addWidget(row(self.s_box, self.c_box, self.t_box, self.b_stream))
        # ---- clocks
        layout.addWidget(label('Clocks (figures from their sources; blank: not stated)', bold=True))
        self.clock_name = QLineEdit()
        self.clock_name.setPlaceholderText('name')
        self.clock_kind = QComboBox()
        for k in CLOCK_KINDS:
            self.clock_kind.addItem(KIND_TEXT[k], k)
        self.f_off = QLineEdit()
        self.f_off.setPlaceholderText('offset ms')
        self.f_jit = QLineEdit()
        self.f_jit.setPlaceholderText('jitter ± ms')
        self.f_drift = QLineEdit()
        self.f_drift.setPlaceholderText('drift ppm')
        for w in (self.f_off, self.f_jit, self.f_drift):
            w.setMaximumWidth(100)
        self.f_src = QLineEdit()
        self.f_src.setPlaceholderText('source (a measurement, a datasheet)')
        self.b_clock = button('Add clock', role='ghost', tip='Add this clock to the rig (one undo step)')
        self.b_clock.clicked.connect(self._add_clock_clicked)
        layout.addWidget(row(self.clock_name, self.clock_kind, self.f_off, self.f_jit, self.f_drift))
        layout.addWidget(row(self.f_src, self.b_clock))
        self.clock_list = QComboBox()
        self.b_clock_del = button('Remove clock', role='ghost', tip='Remove the chosen clock (one undo step)')
        self.b_clock_del.clicked.connect(lambda: self.remove_clock(self.clock_list.currentData()))
        layout.addWidget(row(self.clock_list, self.b_clock_del))
        st = self.ctx.store
        st.changed.connect(lambda _s: self._sync_settings())
        st.project_replaced.connect(self._replaced)
        self._sync_settings()

    # ---------------------------------------------------------------- settings
    def settings(self) -> dict:
        try:
            return TS.settings_of(self.ctx.store.project)
        except Exception:                               # noqa: BLE001 — a bad stored value: the defaults
            traceback.print_exc()
            return dict(TS.DEFAULTS, defaults=list(TS.DEFAULTS), pairing='interpolate', joint_rate_hz=None,
                        latency_s={})

    def _sync_settings(self):
        st = self.settings()
        self.limit_spin.set_value(float(st['skew_limit_s']) * 1000.0)
        self.limit_def.setText('default' if 'skew_limit_s' in st['defaults'] else
                               f'yours (default {TS.DEFAULTS["skew_limit_s"] * 1000:g} ms)')
        self.ep_spin.set_value(float(st['episode_s']))
        self.ep_def.setText('default' if 'episode_s' in st['defaults'] else
                            f'yours (default {TS.DEFAULTS["episode_s"]:g} s)')
        self.pair_box.setCurrentIndex(max(0, self.pair_box.findData(st['pairing'])))
        jr = st['joint_rate_hz']
        self.rate_edit.setText('' if jr is None else f'{jr["value"]:g}')
        self.rate_src.setText('' if jr is None else jr['source'])
        p = self.ctx.store.project
        cfg = TS.sync_config(p)
        cur = self.s_box.currentData()
        self.s_box.blockSignals(True)
        self.s_box.clear()
        for s in p.variant.sensors:
            self.s_box.addItem(s.name, s.uid)
        self.s_box.addItem('joint_states', 'joint_states')
        k = self.s_box.findData(cur)
        self.s_box.setCurrentIndex(k if k >= 0 else 0)
        self.s_box.blockSignals(False)
        self.c_box.clear()
        self.c_box.addItem('host arrival (default)', '')
        self.clock_list.clear()
        for c in cfg['clocks']:
            self.c_box.addItem(f'{c["name"]} ({KIND_TEXT.get(c["kind"], c["kind"])})', c['uid'])
            self.clock_list.addItem(c['name'], c['uid'])
        self.b_clock_del.setEnabled(bool(cfg['clocks']))

    def _set_setting(self, key, value, text):
        if self.settings().get(key) == value:
            return

        def fn(p):
            d = dict(p.analysis.studies.get(TS.STUDY_ID) or {})
            d[key] = value
            p.analysis.studies[TS.STUDY_ID] = d
        self.ctx.store.edit('analysis', fn, text, merge_key=(TS.STUDY_ID, key))

    def set_joint_rate(self, hz, source: str) -> bool:
        if hz is not None and hz <= 0:
            if self.ctx.window is not None:
                self.ctx.window.msg('The joint_states rate must be a positive number of hertz.', 6000)
            return False

        def fn(p):
            d = dict(p.analysis.studies.get(TS.STUDY_ID) or {})
            if hz is None:
                d.pop('joint_rate_hz', None)
            else:
                d['joint_rate_hz'] = {'value': float(hz), 'source': str(source or '').strip()}
            p.analysis.studies[TS.STUDY_ID] = d
        self.ctx.store.edit('analysis', fn, 'Time sync · joint_states rate' if hz is not None else
                            'Time sync · joint_states rate cleared')
        return True

    def _edit_sync(self, fn, text):
        """Change project.rig['sync'] through fn(cfg) (one undo step)."""
        from ...core.project import rig_from_dict, sync_from_dict

        def edit(p):
            rig = rig_from_dict(p.rig)
            cfg = sync_from_dict(rig.get('sync'))
            fn(cfg)
            rig['sync'] = cfg
            p.rig = rig_from_dict(rig)
        self.ctx.store.edit('rig', edit, text)

    def set_stream(self, key, clock_uid, trigger) -> bool:
        """Set a stream's clock and trigger (one undo step). A hardware trigger on a device whose maker states none is
        refused with the maker's words."""
        if not key:
            return False
        p = self.ctx.store.project
        s = p.variant.sensor(key) if key != 'joint_states' else None
        if trigger == 'hardware' and s is not None:
            from ...core import library as LB
            e = LB.get(s.model_id) if s.model_id else None
            sy = (e.data.get('sync') if e is not None else None) or {}
            if isinstance(sy, dict) and sy.get('hardware_trigger') is False:
                if self.ctx.window is not None:
                    why = sy.get('what') or 'its maker says so'
                    self.ctx.window.msg(f'{s.name} cannot take a hardware trigger: {why}', 10000)
                return False

        def fn(cfg):
            cfg['streams'][key] = {'clock': clock_uid or '', 'trigger': trigger or 'free'}
        name = s.name if s is not None else 'joint_states'
        self._edit_sync(fn, f'Time sync · {name}: {TRIGGER_TEXT.get(trigger, trigger)}')
        return True

    def _add_clock_clicked(self):
        src = self.f_src.text().strip()
        figs = {}
        for key, w, scale in (('offset_s', self.f_off, 1e-3), ('jitter_s', self.f_jit, 1e-3),
                              ('drift_ppm', self.f_drift, 1.0)):
            v = _num(w.text())
            figs[key] = None if v is None else {'value': v * scale, 'source': src}
        self.add_clock(self.clock_name.text().strip() or 'Clock', self.clock_kind.currentData(), figs)

    def add_clock(self, name: str, kind: str, figures: dict | None = None) -> str:
        """Add a clock to the rig (one undo step); its uid."""
        uid = new_uid()
        f = figures or {}

        def fn(cfg):
            cfg['clocks'].append({'uid': uid, 'name': name, 'kind': kind, 'offset_s': f.get('offset_s'),
                                  'jitter_s': f.get('jitter_s'), 'drift_ppm': f.get('drift_ppm')})
        self._edit_sync(fn, f'Time sync · add clock {name}')
        return uid

    def remove_clock(self, uid) -> bool:
        if not uid:
            return False

        def fn(cfg):
            cfg['clocks'] = [c for c in cfg['clocks'] if c['uid'] != uid]
            for v in cfg['streams'].values():
                if v.get('clock') == uid:
                    v['clock'] = ''
        self._edit_sync(fn, 'Time sync · remove clock')
        return True

    def _replaced(self):
        self.doc = None
        self._overlay3d = None
        for w in (self.streams, self.budget, self.pairs):
            w.clear()
        for w in (self.missing, self.joint_label, self.sugg_label, self.notes_label):
            w.setText('')
        self._sync_settings()
        self.suggestion_ready(False)
        self.empty()

    # ---------------------------------------------------------------- result
    def _headline(self, doc):
        s = doc.get('summary') or {}
        sev, word = s.get('status', 'warning'), s.get('word', '')
        lim = doc.get('skew_limit_s', TS.DEFAULTS['skew_limit_s'])
        tip = (f'Skew limit {lim * 1000:g} ms (' + ('the default' if doc.get('skew_limit_default') else
                                                    'your setting') + '): within it "in sync", up to twice "tight", '
               'beyond "out of sync"; "at least" or "incomplete" when a term is not stated.')
        w = next((x for x in doc.get('streams', []) if x['key'] == s.get('worst')), None)
        if w is None:
            self.head.set('—', '', sev, word, '', tip)
            return
        lo, hi = s.get('skew_interval_s') or [None, None]
        text = (f'{w["name"]} ({w["clock"]}, {TRIGGER_TEXT.get(w["trigger"], w["trigger"])}): its stamps are '
                f'{_ms(lo)} to {_ms(hi)} ms off joint_states.')
        le = s.get('label_error_worst')
        if le:
            text += (f' The end effector moves up to {_mm(le["error_m"])} mm in that time on {le["trajectory"]} '
                     '(the action-label error).')
        if not s.get('complete'):
            text += ' Some terms are not stated (see the budget), so the skew can be larger.'
        self.head.set(('≥ ' if not s.get('complete') else '') + _ms(s.get('worst_skew_s')), 'ms', sev, word, text, tip)

    def show_result(self, doc):
        self._headline(doc)
        cur = self._selected_key()
        self.streams.clear()
        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        for x in doc.get('streams', []):
            it = QTreeWidgetItem([x['name'], x['clock'], TRIGGER_TEXT.get(x['trigger'], x['trigger']),
                                  _yes(x['hardware_trigger']), f'{_ms(x["skew_low_s"])} … {_ms(x["skew_high_s"])}',
                                  _ms(x['spread_s']), _mm(x['label_error_m']), x['word']])
            it.setData(0, Qt.ItemDataRole.UserRole, x['key'])
            it.setToolTip(3, x['hardware_trigger_what'])
            it.setToolTip(5, 'The width of the skew interval: what is left if a constant latency is subtracted')
            it.setToolTip(6, 'The end effector\'s largest displacement during the skew along the trajectories')
            for k in (4, 5, 6):
                it.setFont(k, mono(11))
                it.setTextAlignment(k, right)
            it.setForeground(7, QColor(STATUS[SEV_COLOUR.get(x['status'], 'warning')]))
            self.streams.addTopLevelItem(it)
            if x['key'] == cur:
                self.streams.setCurrentItem(it)
        if self.streams.currentItem() is None and self.streams.topLevelItemCount():
            key = (doc.get('summary') or {}).get('worst')
            items = [self.streams.topLevelItem(i) for i in range(self.streams.topLevelItemCount())]
            self.streams.setCurrentItem(next((i for i in items if i.data(0, Qt.ItemDataRole.UserRole) == key),
                                             items[0]))
        _fit(self.streams)
        self._fill_budget()
        js = doc.get('joint_states') or {}
        how = {'interpolate': 'labels interpolated', 'nearest': 'labels from the nearest sample'}.get(
            js.get('pairing'), '')
        self.joint_label.setText(f'joint_states: {js.get("clock", "")}, {how}; stamp error '
                                 f'{_ms(js.get("low_s"))} … {_ms(js.get("high_s"))} ms'
                                 + (f'; not stated: {"; ".join(js.get("not_stated") or [])}' if js.get('not_stated')
                                    else '') + '.')
        self.pairs.clear()
        for pr in doc.get('pairs', []):
            it = QTreeWidgetItem([f'{pr["a"]} / {pr["b"]}', _ms(pr.get('phase_s')), _ms(pr.get('worst_s')),
                                  'yes' if pr.get('locked') else 'no', 'yes' if pr.get('cable_possible') else 'no'])
            for k in (1, 2):
                it.setFont(k, mono(11))
                it.setTextAlignment(k, right)
            self.pairs.addTopLevelItem(it)
        _fit(self.pairs)
        sg = doc.get('suggestion')
        self.sugg_label.setText(('Suggestion: ' + sg['text']) if sg else 'Every stream is in sync and its budget '
                                                                          'complete.')
        self.suggestion_ready(bool(sg and sg.get('kind') in ('trigger', 'sync_cable')))
        self.notes_label.setText('\n'.join(doc.get('notes') or []))
        self._redraw()

    def _selected_key(self):
        it = self.streams.currentItem()
        return it.data(0, Qt.ItemDataRole.UserRole) if it is not None else None

    def _stream(self):
        key = self._selected_key()
        return next((x for x in (self.doc or {}).get('streams', []) if x['key'] == key), None)

    def _fill_budget(self):
        self.budget.clear()
        x = self._stream()
        if x is None:
            self.missing.setText('')
            _fit(self.budget)
            return
        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        for t in x['stamp']['terms']:
            it = QTreeWidgetItem([t['term'], _ms(t['low_s'], 3), _ms(t['high_s'], 3), t['source']])
            it.setToolTip(3, t['source'])
            for k in (1, 2):
                it.setFont(k, mono(11))
                it.setTextAlignment(k, right)
            self.budget.addTopLevelItem(it)
        _fit(self.budget)
        ns = x['stamp']['not_stated']
        self.missing.setText(('Not counted (not stated): ' + '; '.join(ns) + '.') if ns else 'Every term is stated.')
        self.b_worst.setEnabled(bool(x.get('label_error_worst')))
        self._overlay3d = None

    # ---------------------------------------------------------------- the worst moment
    def show_worst(self) -> bool:
        """Pose the robot where the selected stream's action-label error is largest (a trajectory of the project)."""
        from ...core.robot import trajectory as TR
        from ...core.robot.model import get_model
        x = self._stream()
        w = (x or {}).get('label_error_worst')
        if not w:
            return False
        p = self.ctx.store.project
        tr = next((t for t in p.tasks.trajectories if isinstance(t, dict) and t.get('uid') == w['trajectory_uid']),
                  None)
        if tr is None:
            if self.ctx.window is not None:
                self.ctx.window.msg('That motion is a grasp approach, not a trajectory of the project: its pose is not '
                                    'kept.', 6000)
            return False
        try:
            m = get_model(p, tr['robot_uid'])
            self.ctx.store.set_q(tr['robot_uid'], TR.on_model(tr, m.joint_names, m.home, float(w['t'])))
        except Exception:                               # noqa: BLE001 — said, logged
            traceback.print_exc()
            return False
        self._redraw()
        return True

    # ---------------------------------------------------------------- overlays
    def _segment(self):
        w = (self._stream() or {}).get('label_error_worst')
        if not w or not w.get('tcp_at_exposure'):
            return None
        return np.array([w['tcp_at_exposure'], w['tcp_at_stamp']], float)

    def overlays(self, vp) -> list:
        if not self.show3d.isChecked() or self.doc is None:
            return []
        if self._overlay3d is not None:
            return self._overlay3d
        from ..viewport3d import LineDraw
        items = []
        for tr in (self.doc.get('motions') or {}).get('trajectories', []):
            P = np.asarray(tr.get('tcp_path') or [], float)
            if len(P) >= 2:
                items.append(LineDraw(P.astype(np.float32), QColor(THEME.color('ink2')), 1.5, depth=False))
        seg = self._segment()
        x = self._stream()
        if seg is not None and x is not None:
            items.append(LineDraw(seg.astype(np.float32), QColor(STATUS[SEV_COLOUR.get(x['status'], 'warning')]),
                                  4.0, depth=False))
        self._overlay3d = items
        return items

    def camview_overlays(self, cam_model, pose) -> list:
        from ...core import camrender as CR
        seg = self._segment()
        x = self._stream()
        if seg is None or x is None or not self.show3d.isChecked():
            return []
        uv, ok, _d = CR.project_world(cam_model, pose, seg)
        if not ok.all():
            return []
        w = x['label_error_worst']
        return [([uv], QColor(STATUS[SEV_COLOUR.get(x['status'], 'warning')]),
                 f'action-label error {_mm(w["error_m"])} mm at t = {w["t"]:.2f} s ({x["name"]})')]

    def _redraw(self):
        self._overlay3d = None
        win = self.ctx.window
        if win is not None and hasattr(win, 'viewport'):
            win.viewport.update()
        cv = getattr(win, 'camview', None) if win is not None else None
        if cv is not None and hasattr(cv, 'view'):
            cv.view.refresh_overlays()

    # ---------------------------------------------------------------- apply / preview / explain
    def apply(self):
        """Make the suggested trigger or sync cable (one undo step), then run again."""
        sg = (self.doc or {}).get('suggestion')
        if not sg or sg.get('kind') not in ('trigger', 'sync_cable'):
            return False
        text = (f'Time sync · hardware trigger for {sg.get("sensor", "")}' if sg['kind'] == 'trigger' else
                'Time sync · sync cable')
        self.ctx.store.edit('rig', lambda p: TS.apply_suggestion(p, sg), text)
        self.suggestion_ready(False)
        self.run()
        return True

    def preview(self):
        sg = (self.doc or {}).get('suggestion')
        if sg and self.ctx.window is not None:
            self.ctx.window.msg('Preview: ' + sg['text'] + (' Apply makes the change (one undo step).'
                                                            if sg.get('kind') in ('trigger', 'sync_cable') else ''),
                                15000)
        return sg

    def explain_text(self) -> str:
        st = self.settings()
        return ('Each camera frame\'s stamp is compared with the instant it was exposed (the middle of the middle '
                'row\'s exposure). Stamped by the host on arrival, the frame is late by half the exposure, the '
                'readout, its USB transfer (at the link\'s capacity, at worst behind one frame of every other device '
                'on the controller: the Bandwidth page\'s topology) and up to one 125 µs bus interval; the device\'s '
                'own processing is not stated by the makers, so that budget is a lower bound until you enter a '
                'measured latency. A hardware trigger makes the stamp the trigger edge; a device clock adds its '
                'stated offset, jitter and drift. joint_states has its own budget. The skew against joint_states, '
                'applied to the end effector\'s motion (forward kinematics along the trajectory), is the '
                f'action-label error.\n\nThe skew limit is {st["skew_limit_s"] * 1000:g} ms (default 2 ms, a '
                'DexVantage default, not a standard). PTP and NTP figures are never assumed: enter what your servo '
                'reports, with its source.\n\nKnown limits: isochronous reservations and driver buffering are not '
                'modelled; the trigger-to-exposure delay is not counted; other joints stay at home.')

    def explain(self):
        box = QMessageBox(self)
        box.setWindowTitle('How the time sync budget is computed')
        box.setText(self.explain_text())
        box.setModal(False)
        box.show()


def _columns(tree: QTreeWidget):
    h = tree.header()
    h.setStretchLastSection(False)
    h.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    for k in range(1, tree.columnCount()):
        h.setSectionResizeMode(k, QHeaderView.ResizeMode.ResizeToContents)


def _fit(tree: QTreeWidget):
    n = tree.topLevelItemCount()
    rh = tree.sizeHintForRow(0) if n else 20
    tree.setFixedHeight(tree.header().sizeHint().height() + max(1, n) * max(rh, 18) + 6)


register('time_sync', 'Time sync', TimeSyncPage, menu='Analysis', icon='sync', workspace='placement', order=66)
