"""The Cable flex page (spec §7.5): whether each cable survives the joints it crosses.

What it shows: the number of cables that keep their length, loop and bend radius over the joint ranges and the
trajectories (the headline, with a status word: "clear", "check" when a figure is missing or a laid length is not
set, "at risk" for a snag or a bend below the allowed radius, "pulls" when a run is laid too short), the attachment
editor (the cable's links and points, its type from the library, its stated minimum bend radius with its source, the
laid length of each run), a table per run (the joints crossed, the slack loop each needs, the length range, the
suggested and laid lengths, the smallest bend radius) with the flags in plain words, and the cable polylines drawn in
the 3D view and the Camera view at the joint state on screen, each run in its status colour. Apply sets every run's
laid length to the suggested one (one undo step); a project without cables gets suggested routes, which Apply adds.
Export writes the harness sheet as JSON (with provenance), CSV or HTML.
"""
from __future__ import annotations

import os
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QGridLayout, QHeaderView, QLineEdit, QMessageBox,
                             QTableWidget, QTreeWidget, QTreeWidgetItem, QWidget)

from ...core import cables as CB
from ...core.studies import cables as CS
from ..theme import STATUS, mono
from ..widgets import ScrubSpinBox, button, label
from . import register
from .base import StudyPage, row

SETTINGS = (('bend_factor', 'Bend factor', '×', 'The allowed bend radius is the stated minimum times this (default '
                                                '1.5, a DexVantage margin for flexing, not a maker\'s figure).'),
            ('loop_m', 'Service loop', 'm', 'Added to a run\'s largest length for the suggested laid length '
                                            '(default 0.05 m).'))


class _PolylineSignal(QObject):
    done = pyqtSignal(int, object)                     # (generation, [(points, status)]) from the worker thread


def _mm(v, d=1) -> str:
    return '—' if v is None else f'{v * 1000:.{d}f}'


def _m(v, d=3) -> str:
    return '—' if v is None else f'{v:.{d}f}'


class CablesPage(StudyPage):
    """Cable flex across joints: each run between two attachment points is swept over the full range of the joints it
    crosses (the model file's limits) and along the trajectories; the length it needs, the slack loop per joint, the
    loop's bend radius against the cable's stated minimum × the bend factor, and snags of the loop on the links."""
    STUDY = 'cables'
    TITLE = 'Cable flex across joints'
    HEADLINE = 'Cables clear over the joint ranges'

    # ---------------------------------------------------------------- body
    def build_body(self, layout):
        self._overlay_cache = None
        self._overlay_last = []                         # the last polylines computed, drawn while new ones compute
        self._overlay_gen = 0                           # bumped by every change that makes the cache stale
        self._overlay_running = False
        self._overlay_sig = _PolylineSignal()
        self._overlay_sig.done.connect(self._polylines_ready)
        self._preview = False
        self._busy = False
        self._spin = {}
        layout.addWidget(label('Cable', bold=True))
        self.cable_box = QComboBox()
        self.cable_box.setMinimumWidth(220)
        self.cable_box.setToolTip('The cable to edit')
        self.cable_box.setPlaceholderText('No cable yet: New cable adds one')
        self.b_new = button('New cable', role='ghost', tip='Add a cable for the selected sensor (one undo step)')
        self.b_del = button('Delete', role='ghost', tip='Delete this cable (one undo step)')
        self.b_suggest = button('Suggest attachments', role='ghost',
                                tip='Place one attachment per link from the sensor to the base, so each run crosses '
                                    'one joint (a suggestion to check against the real robot)')
        layout.addWidget(row(self.cable_box, self.b_new, self.b_del, self.b_suggest))
        self.none_hint = label('', muted=True, wrap=True)
        layout.addWidget(self.none_hint)
        g = QWidget()
        grid = QGridLayout(g)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        self.name_edit = QLineEdit()
        self.sensor_box = QComboBox()
        self.sensor_box.setToolTip('The sensor this cable serves')
        self.lib_box = QComboBox()
        self.lib_box.setToolTip('A cable from the library: its type, diameter, connectors and minimum bend radius come '
                                'from the maker\'s datasheet')
        self.dia = ScrubSpinBox('', 'plain', 2, 'mm', step=0.1, width=110)
        self.dia.setRange(0.0, 100.0)
        self.dia.setToolTip('Outer diameter; 0 = not stated')
        self.bend = ScrubSpinBox('', 'plain', 1, 'mm', step=1.0, width=110)
        self.bend.setRange(0.0, 2000.0)
        self.bend.setToolTip('The maker\'s minimum bend radius; 0 = not stated (bends are then not checked)')
        self.bend_src = label('', muted=True, small=True, wrap=True)
        self.length = ScrubSpinBox('', 'plain', 3, 'm', step=0.05, width=110)
        self.length.setRange(0.0, 1000.0)
        self.length.setToolTip('The cable\'s provided length; 0 = not set')
        self.wrap = ScrubSpinBox('', 'plain', 1, 'mm', step=1.0, width=110)
        self.wrap.setRange(0.0, 1000.0)
        self.wrap.setToolTip('The radius the cable is guided round a joint at; 0 = the default (the nearer '
                             'attachment\'s distance from the axis)')
        for sb, text in ((self.dia, '—'), (self.bend, '—'), (self.length, '—'), (self.wrap, 'default')):
            sb.setSpecialValueText(text)                # 0 is "not stated" (or the default), never a figure
        self.conn = QLineEdit()
        self.conn.setPlaceholderText('e.g. FAKRA-Z female / FAKRA-Z female')
        self.relief = QLineEdit()
        self.relief.setPlaceholderText('e.g. P-clip at each attachment, 20 mm boot at the camera')
        for i, (text, w) in enumerate((('Name', self.name_edit), ('Sensor', self.sensor_box),
                                       ('Library cable', self.lib_box), ('Diameter', self.dia),
                                       ('Minimum bend radius', self.bend), ('Provided length', self.length),
                                       ('Wrap radius', self.wrap), ('Connectors', self.conn),
                                       ('Strain relief', self.relief))):
            grid.addWidget(label(text), i, 0)
            grid.addWidget(w, i, 1)
        grid.addWidget(self.bend_src, 4, 2)
        grid.setColumnStretch(2, 1)
        layout.addWidget(g)
        layout.addWidget(label('Attachment points (link frame, mm; laid length to the next point)', bold=True))
        self.att = QTableWidget(0, 5)
        self.att.setHorizontalHeaderLabels(['Link', 'x', 'y', 'z', 'Laid to next'])
        self.att.verticalHeader().setVisible(False)
        h = self.att.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for k in range(1, 5):
            h.setSectionResizeMode(k, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.att)
        self.b_add = button('Add point', role='ghost', tip='Add an attachment point after the selected one')
        self.b_remove = button('Remove point', role='ghost', tip='Remove the selected attachment point')
        self.b_sensor_pt = button('Use the sensor\'s position', role='ghost',
                                  tip='Set the selected point to the sensor\'s mount position on its link')
        layout.addWidget(row(self.b_add, self.b_remove, self.b_sensor_pt))
        layout.addWidget(label('Runs', bold=True))
        self.runs = QTreeWidget()
        self.runs.setRootIsDecorated(False)
        self.runs.setHeaderLabels(['Run', 'Joints', 'Slack loop per joint', 'Length min–max', 'Suggested', 'Laid',
                                   'Smallest bend', 'Status'])
        h = self.runs.header()
        h.setStretchLastSection(False)
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for k in range(1, 8):
            h.setSectionResizeMode(k, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.runs)
        self.flags = label('', wrap=True)
        layout.addWidget(self.flags)
        self.notes_label = label('', muted=True, wrap=True)
        layout.addWidget(self.notes_label)
        self.show3d = QCheckBox('Show cables in the 3D view')
        self.show3d.setChecked(True)
        self.show3d.toggled.connect(lambda _on: self._redraw())
        layout.addWidget(self.show3d)
        layout.addWidget(label('Settings (your settings; defaults labelled)', bold=True))
        sg = QWidget()
        sgrid = QGridLayout(sg)
        sgrid.setContentsMargins(0, 0, 0, 0)
        for i, (key, text, unit, tip) in enumerate(SETTINGS):
            sb = ScrubSpinBox('', 'plain', 2 if unit == '×' else 3, unit, step=0.1 if unit == '×' else 0.01,
                              width=110)
            lo, hi = CS.LIMITS[key]
            sb.setRange(lo, hi)
            sb.setToolTip(tip)
            dl = label('', muted=True, small=True)
            sgrid.addWidget(label(text), 0, i * 3)
            sgrid.addWidget(sb, 0, i * 3 + 1)
            sgrid.addWidget(dl, 0, i * 3 + 2)
            self._spin[key] = (sb, dl)
            sb.committed.connect(lambda v, k=key: self._set_setting(k, v))
        self.ranges_box = QComboBox()
        self.ranges_box.addItem('The model file\'s limits', 'model')
        self.ranges_box.addItem('The project\'s trajectories', 'trajectories')
        self.ranges_box.setToolTip('The span each crossed joint is swept over. The model file\'s limits (default) size '
                                   'the cable for every turn the joint can make; the trajectories size it for what the '
                                   'task does (home included).')
        self.ranges_default = label('', muted=True, small=True)
        sgrid.addWidget(label('Joint ranges'), 1, 0)
        sgrid.addWidget(self.ranges_box, 1, 1, 1, 2)
        sgrid.addWidget(self.ranges_default, 1, 3)
        self.ranges_box.activated.connect(lambda i: self._set_ranges(self.ranges_box.itemData(i)))
        layout.addWidget(sg)
        self.stale_note = label('', muted=True, wrap=True)
        layout.addWidget(self.stale_note)
        # wiring
        self.cable_box.currentIndexChanged.connect(lambda _i: self._fill_cable())
        self.b_new.clicked.connect(self.new_cable)
        self.b_del.clicked.connect(self.delete_cable)
        self.b_suggest.clicked.connect(self.suggest)
        self.b_add.clicked.connect(self.add_point)
        self.b_remove.clicked.connect(self.remove_point)
        self.b_sensor_pt.clicked.connect(self.sensor_point)
        self.name_edit.editingFinished.connect(lambda: self._set_field('name', self.name_edit.text().strip()
                                                                       or 'Cable'))
        self.sensor_box.activated.connect(lambda i: self._set_field('sensor_uid', self.sensor_box.itemData(i) or ''))
        self.lib_box.activated.connect(lambda i: self.set_library(self.lib_box.itemData(i) or ''))
        self.dia.committed.connect(lambda v: self._set_field('diameter_m', v / 1000.0 if v > 0 else None))
        self.bend.committed.connect(self._set_bend)
        self.length.committed.connect(lambda v: self._set_field('length_m', v if v > 0 else None))
        self.wrap.committed.connect(lambda v: self._set_field('wrap_radius_m', v / 1000.0 if v > 0 else None))
        self.conn.editingFinished.connect(lambda: self._set_field(
            'connectors', [c.strip() for c in self.conn.text().split('/') if c.strip()]))
        self.relief.editingFinished.connect(lambda: self._set_field('strain_relief', self.relief.text().strip()))
        st = self.ctx.store
        st.changed.connect(self._changed)
        st.pose_changed.connect(lambda _uid: self._invalidate())
        st.project_replaced.connect(self._replaced)
        self._refresh_all()

    # ---------------------------------------------------------------- project access
    @property
    def project(self):
        return self.ctx.store.project

    def cables(self) -> list:
        return [CB.clean_cable(c) for c in getattr(self.project, 'cables', []) or []]

    def current(self) -> dict | None:
        uid = self.cable_box.currentData()
        return next((c for c in self.cables() if c['uid'] == uid), None)

    def _edit(self, fn, text, merge_key=None):
        """fn(list of cable records) changes them in place; one undo step on the 'cables' section."""
        def apply(p):
            cs = [CB.clean_cable(c) for c in getattr(p, 'cables', []) or []]
            fn(cs)
            p.cables = cs
        self.ctx.store.edit('cables', apply, text, merge_key=merge_key)

    def _edit_current(self, fn, text, merge_key=None):
        cur = self.current()
        if cur is None:
            return
        uid = cur['uid']

        def f(cs):
            for c in cs:
                if c['uid'] == uid:
                    fn(c)
        self._edit(f, f'{cur["name"]} · {text}', merge_key=merge_key)

    # ---------------------------------------------------------------- refresh
    def _changed(self, sections):
        if not sections or {'cables', 'variants', 'robots'} & set(sections) or any(
                str(s).startswith('variant') for s in sections):
            self._refresh_all()
        if 'analysis' in (sections or ()):
            self._sync_settings()
        self._invalidate()
        self._check_stale()

    def _replaced(self):
        self.doc = None
        self.runs.clear()
        self.flags.setText('')
        self.notes_label.setText('')
        self.stale_note.setText('')
        self._refresh_all()
        self.empty()
        self._invalidate()

    def _refresh_all(self):
        self._busy = True
        try:
            cur = self.cable_box.currentData()
            self.cable_box.blockSignals(True)
            self.cable_box.clear()
            for c in self.cables():
                self.cable_box.addItem(c['name'], c['uid'])
            k = self.cable_box.findData(cur)
            self.cable_box.setCurrentIndex(k if k >= 0 else (0 if self.cable_box.count() else -1))
            self.cable_box.blockSignals(False)
            self.sensor_box.clear()
            self.sensor_box.addItem('No sensor', '')
            for s in self.project.variant.sensors:
                self.sensor_box.addItem(s.name, s.uid)
            self.lib_box.clear()
            self.lib_box.addItem('No library cable (your figures)', '')
            try:
                from ...core import library
                for e in library.entries('cable'):
                    self.lib_box.addItem(f'{e.maker} {e.name}' + (' (custom)' if e.is_custom else ''), e.id)
            except Exception:                           # noqa: BLE001 — a broken library must not take the page
                traceback.print_exc()
            self._sync_settings()
        finally:
            self._busy = False
        self._fill_cable()

    def _fill_cable(self):
        c = self.current()
        has = c is not None
        for w in (self.b_del, self.b_suggest, self.name_edit, self.sensor_box, self.lib_box, self.dia, self.bend,
                  self.length, self.wrap, self.conn, self.relief, self.att, self.b_add, self.b_remove,
                  self.b_sensor_pt):
            w.setEnabled(has)
        self.att.setRowCount(0)
        sug = (self.doc or {}).get('suggested_cables') or []
        if has:
            self.none_hint.setText('')
        elif sug:
            self.none_hint.setText(f'The project has no cables yet. The result above is for a suggested route '
                                   f'({", ".join(c["name"] for c in sug)}), drawn in the 3D view: Apply adds it as a '
                                   'cable you can then edit here, or New cable starts one of your own.')
        else:
            self.none_hint.setText('The project has no cables yet: New cable adds one for the selected sensor.')
        if not has:
            self.bend_src.setText('')
            return
        self._busy = True
        try:
            self.name_edit.setText(c['name'])
            self.sensor_box.setCurrentIndex(max(0, self.sensor_box.findData(c['sensor_uid'])))
            k = self.lib_box.findData(c['library_id'])
            if c['library_id'] and k < 0:
                self.lib_box.addItem(f'{c["library_id"]} (not in the library)', c['library_id'])
                k = self.lib_box.count() - 1
            self.lib_box.setCurrentIndex(max(0, k))
            self.dia.set_value((c['diameter_m'] or 0.0) * 1000.0)
            mb = c['min_bend_radius_m']
            self.bend.set_value((mb['value'] if mb else 0.0) * 1000.0)
            self.bend_src.setText(('Source: ' + mb['source']) if mb else CB.NOT_STATED)
            self.length.set_value(c['length_m'] or 0.0)
            self.wrap.set_value((c['wrap_radius_m'] or 0.0) * 1000.0)
            self.conn.setText(' / '.join(c['connectors']))
            self.relief.setText(c['strain_relief'])
            links = self._links(c)
            for i, a in enumerate(c['attachments']):
                self.att.insertRow(i)
                box = QComboBox()
                for ln in links:
                    box.addItem(ln)
                if a['link'] and box.findText(a['link']) < 0:
                    box.addItem(a['link'])
                box.setCurrentIndex(max(0, box.findText(a['link'])))
                box.activated.connect(lambda _j, i=i, b=box: self._set_att(i, 'link', b.currentText()))
                self.att.setCellWidget(i, 0, box)
                for k_, axis in enumerate('xyz'):
                    sb = ScrubSpinBox('', 'plain', 1, 'mm', step=1.0, width=90)
                    sb.setRange(-5000.0, 5000.0)
                    sb.set_value(a['point'][k_] * 1000.0)
                    sb.committed.connect(lambda v, i=i, k_=k_: self._set_point(i, k_, v / 1000.0))
                    self.att.setCellWidget(i, 1 + k_, sb)
                sb = ScrubSpinBox('', 'plain', 3, 'm', step=0.01, width=100)
                sb.setRange(0.0, 100.0)
                sb.setSpecialValueText('—')             # 0: not set, never a figure
                sb.set_value(a['length_to_next_m'] or 0.0)
                sb.setToolTip('The cable laid between this point and the next; 0 = not set (the suggested length is '
                              'used)')
                sb.setEnabled(i < len(c['attachments']) - 1)
                sb.committed.connect(lambda v, i=i: self._set_att(i, 'length_to_next_m', v if v > 0 else None))
                self.att.setCellWidget(i, 4, sb)
            n = max(1, self.att.rowCount())
            self.att.setFixedHeight(self.att.horizontalHeader().sizeHint().height() + n * 30 + 6)
        finally:
            self._busy = False

    def _links(self, c) -> list:
        ruid = next((a['robot_uid'] for a in c['attachments'] if a['robot_uid']), '') or self._robot_uid(c)
        try:
            from ...core.robot.model import get_model
            return list(get_model(self.project, ruid).link_names) if ruid else []
        except Exception:                               # noqa: BLE001 — a robot that cannot compile: no links
            traceback.print_exc()
            return []

    def _robot_uid(self, c) -> str:
        s = self.project.variant.sensor(c['sensor_uid']) if c and c['sensor_uid'] else None
        if s is not None and s.mount.on_link:
            return s.mount.robot_uid
        r = self.project.primary_robot
        return r.uid if r is not None else ''

    # ---------------------------------------------------------------- edits
    def _set_field(self, key, value):
        if self._busy:
            return
        cur = self.current()
        if cur is None or cur.get(key) == value:
            return

        def fn(c):
            c[key] = value
        self._edit_current(fn, f'{key.replace("_m", "").replace("_", " ")}', merge_key=('cables', key))

    def _set_bend(self, v):
        if self._busy:
            return
        cur = self.current()
        if cur is None:
            return
        old = cur['min_bend_radius_m']
        new = None if v <= 0 else {'value': v / 1000.0, 'source': 'entered by you', 'kind': 'unspecified'}
        if old is not None and new is not None and abs(old['value'] - new['value']) < 1e-12:
            return

        def fn(c):
            c['min_bend_radius_m'] = new
        self._edit_current(fn, 'minimum bend radius')

    def set_library(self, lib_id: str):
        if self._busy:
            return
        from ...core import library
        e = library.get(lib_id) if lib_id else None

        def fn(c):
            if e is None:
                c['library_id'] = ''
            else:
                c.update(CB.from_library(e))
        self._edit_current(fn, 'library cable ' + (e.name if e is not None else 'none'))

    def _set_att(self, i, key, value):
        if self._busy:
            return

        def fn(c):
            if i < len(c['attachments']):
                c['attachments'][i][key] = value
        self._edit_current(fn, f'point {i + 1} {key.replace("_", " ")}', merge_key=('cables', i, key))

    def _set_point(self, i, k, value):
        if self._busy:
            return

        def fn(c):
            if i < len(c['attachments']):
                c['attachments'][i]['point'][k] = float(value)
        self._edit_current(fn, f'point {i + 1} {"xyz"[k]}', merge_key=('cables', i, k))

    def new_cable(self):
        sensors = [s for s in self.project.variant.sensors if s.mount.on_link] or list(self.project.variant.sensors)
        s = next((x for x in sensors if x.uid == self.sensor_box.currentData()), sensors[0] if sensors else None)
        c = CB.new_cable(f'{s.name} cable' if s is not None else 'Cable', s.uid if s is not None else '')
        self._edit(lambda cs: cs.append(c), f'New cable {c["name"]}')
        k = self.cable_box.findData(c['uid'])
        if k >= 0:
            self.cable_box.setCurrentIndex(k)
        self.suggest()

    def delete_cable(self):
        cur = self.current()
        if cur is None:
            return
        self._edit(lambda cs: cs.__setitem__(slice(None), [c for c in cs if c['uid'] != cur['uid']]),
                   f'Delete cable {cur["name"]}')

    def suggest(self):
        cur = self.current()
        if cur is None:
            return
        s = self.project.variant.sensor(cur['sensor_uid']) if cur['sensor_uid'] else None
        if s is None or not s.mount.on_link:
            self._say('Suggest attachments needs a sensor mounted on a robot link: choose the cable\'s sensor first.')
            return
        try:
            from ...core.robot.model import get_model
            atts, notes = CB.suggest_attachments(get_model(self.project, s.mount.robot_uid), s, cur['diameter_m'])
        except Exception as e:                          # noqa: BLE001 — explained, never a traceback
            traceback.print_exc()
            self._say(f'No attachments could be suggested: {e}')
            return

        def fn(c):
            c['attachments'] = atts
        self._edit_current(fn, 'suggested attachments')
        if notes:
            self._say(' '.join(notes))

    def add_point(self):
        cur = self.current()
        if cur is None:
            return
        i = self.att.currentRow()
        i = len(cur['attachments']) - 1 if i < 0 else i
        ruid = next((a['robot_uid'] for a in cur['attachments'] if a['robot_uid']), '') or self._robot_uid(cur)
        links = self._links(cur)
        base = cur['attachments'][i] if cur['attachments'] else {'link': links[0] if links else '',
                                                                 'point': [0.0, 0.0, 0.0]}
        new = {'robot_uid': ruid, 'link': base['link'], 'point': list(base['point']), 'length_to_next_m': None}

        def fn(c):
            c['attachments'].insert(i + 1, new)
        self._edit_current(fn, 'add a point')

    def remove_point(self):
        cur = self.current()
        if cur is None or not cur['attachments']:
            return
        i = self.att.currentRow()
        i = len(cur['attachments']) - 1 if i < 0 else i

        def fn(c):
            if i < len(c['attachments']):
                del c['attachments'][i]
        self._edit_current(fn, f'remove point {i + 1}')

    def sensor_point(self):
        cur = self.current()
        s = self.project.variant.sensor(cur['sensor_uid']) if cur is not None and cur['sensor_uid'] else None
        if s is None or not s.mount.on_link:
            self._say('The cable has no sensor on a robot link to take the position from.')
            return
        i = max(0, self.att.currentRow())
        pt = [float(v) for v in s.mount.pose.position]

        def fn(c):
            a = {'robot_uid': s.mount.robot_uid, 'link': s.mount.parent_frame, 'point': pt,
                 'length_to_next_m': None}
            if i < len(c['attachments']):
                c['attachments'][i].update({k: a[k] for k in ('robot_uid', 'link', 'point')})
            else:
                c['attachments'].append(a)
        self._edit_current(fn, f'point {i + 1} at the sensor')

    def _say(self, text):
        if self.ctx.window is not None:
            self.ctx.window.msg(text, 8000)
        self.note.setText(text)

    # ---------------------------------------------------------------- settings
    def settings(self) -> dict:
        try:
            return CS.settings_of(self.project)
        except Exception:                               # noqa: BLE001 — a bad stored value: show the defaults
            return dict(CS.DEFAULTS, defaults=list(CS.DEFAULTS))

    def _sync_settings(self):
        st = self.settings()
        for key, (sb, dl) in self._spin.items():
            sb.set_value(float(st[key]))
            d = CS.DEFAULTS[key]
            dl.setText('default' if key in st.get('defaults', ()) else f'yours (default {d:g})')
        self.ranges_box.setCurrentIndex(max(0, self.ranges_box.findData(st.get('joint_ranges', 'model'))))
        self.ranges_default.setText('default' if 'joint_ranges' in st.get('defaults', ()) else
                                    'yours (default the model file\'s limits)')

    def _set_ranges(self, value):
        if value == self.settings().get('joint_ranges'):
            return

        def fn(p):
            d = dict(p.analysis.studies.get('cables') or {})
            d['joint_ranges'] = value
            p.analysis.studies['cables'] = d
        src = 'the trajectories' if value == 'trajectories' else 'the model file'
        self.ctx.store.edit('analysis', fn, f'Cables · joint ranges from {src}')

    def _set_setting(self, key, value):
        if abs(float(self.settings()[key]) - float(value)) < 1e-12:
            return

        def fn(p):
            d = dict(p.analysis.studies.get('cables') or {})
            d[key] = float(value)
            p.analysis.studies['cables'] = d
        text = next(t for k, t, _u, _tip in SETTINGS if k == key)
        self.ctx.store.edit('analysis', fn, f'Cables · {text.lower()} {value:g}', merge_key=('cables', key))

    # ---------------------------------------------------------------- result
    def inputs_key(self) -> str:
        import json
        p = self.project
        return json.dumps([getattr(p, 'cables', []), [r.to_dict() for r in p.robots], p.tasks.trajectories,
                           p.analysis.studies.get('cables') or {}], sort_keys=True, default=str)

    def _check_stale(self):
        if self.doc is not None and getattr(self, '_inputs', None) is not None and self.inputs_key() != self._inputs:
            self.stale_note.setText('The cables or the robot changed since this run: Run again to update.')
        else:
            self.stale_note.setText('')

    def show_result(self, doc):
        self._preview = False
        s = doc.get('summary') or {}
        n, clear = s.get('cables', 0), s.get('clear', 0)
        sev = s.get('status', 'warning')
        if not n:
            self.head.set('—', '', 'warning', 'no cables', s.get('advice', ''))
        else:
            tip = ('Clear: no pull, snag or bend below the allowed radius over every joint range and trajectory, and '
                   'every figure stated. ' + ', '.join(f'{k} {v}' for k, v in (s.get('flags') or {}).items()))
            self.head.set(f'{clear} of {n}', 'cables clear', sev, CS.WORDS[sev], s.get('advice', ''), tip)
        self.runs.clear()
        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        msgs = []
        for c in doc.get('cables') or []:
            top = QTreeWidgetItem([c['name'] + (f' · {c["sensor"]}' if c['sensor'] else ''), '', '', '',
                                   _m(c.get('suggested_m')), _m(c.get('length_m')),
                                   f'allowed {_mm(c.get("allowed_bend_m"))}',
                                   CS.WORDS[c['status']]])
            top.setForeground(7, QColor(STATUS[c['status']]))
            top.setToolTip(6, 'Allowed bend radius: the stated minimum × the bend factor')
            top.setToolTip(5, 'The cable\'s provided length')
            self.runs.addTopLevelItem(top)
            for sg in c['segments']:
                it = QTreeWidgetItem([f'  {sg["index"] + 1}. {sg["from"]["link"]} → {sg["to"]["link"]}',
                                      ', '.join(j['name'] for j in sg['joints']) or 'none',
                                      '; '.join(f'{_mm(j["slack_loop_m"], 0)} mm' for j in sg['joints']) or '—',
                                      f'{_m(sg["length_min_m"])}–{_m(sg["length_max_m"])} m',
                                      f'{_m(sg["suggested_m"])} m', _m(sg['laid_m']) + (' m' if sg['laid_m'] else ''),
                                      f'{_mm(sg["bend_radius_min_m"])} mm', CS.WORDS[sg['status']]])
                for k in range(2, 7):
                    it.setFont(k, mono(11))
                    it.setTextAlignment(k, right)
                it.setForeground(7, QColor(STATUS[sg['status']]))
                it.setToolTip(0, f'Route: {sg["route"]}' + (f', wrap radius {_mm(sg.get("wrap_radius_m"))} mm'
                                                            + (' (default)' if sg.get('wrap_radius_default') else '')
                                                            if sg['route'] == 'wrap' else ''))
                it.setToolTip(2, 'Per joint: the largest minus the smallest length over its range, plus the service '
                                 'loop')
                self.runs.addTopLevelItem(it)
                msgs += [(f['severity'], f['message']) for f in sg['flags'] if f['kind'] != 'unset']
            msgs += [(f['severity'], f['message']) for f in c['flags']]
            unset = sum(f['kind'] == 'unset' for sg in c['segments'] for f in sg['flags'])
            if unset:
                msgs.append(('warning', f'{c["name"]}: {unset} run{"s have" if unset > 1 else " has"} no laid length '
                                        'yet; the suggested lengths are used. Apply sets them.'))
        self.flags.setText('\n'.join(f'{CS.WORDS[s_].capitalize()}: {m}' for s_, m in msgs[:12])
                           + (f'\n… and {len(msgs) - 12} more (see Export).' if len(msgs) > 12 else ''))
        notes = list(doc.get('notes') or []) + [n_ for c in doc.get('cables') or [] for n_ in c['notes']] + \
            [n_ for c in doc.get('cables') or [] for sg in c['segments'] for n_ in sg['notes']]
        self.notes_label.setText('\n'.join(dict.fromkeys(notes)))
        can = bool(doc.get('suggested')) or any(sg['laid_m'] != sg['suggested_m'] for c in doc.get('cables') or []
                                                for sg in c['segments'])
        self.suggestion_ready(can)
        self._fill_cable()                              # the empty-project hint names the suggested cable
        self._inputs = self.inputs_key()
        self._check_stale()
        self._invalidate()

    # ---------------------------------------------------------------- Apply, Preview, Explain, Export
    def apply(self):
        if self.doc is None:
            return
        doc = self.doc
        new = CS.apply_suggested(getattr(self.project, 'cables', []) or [], doc)

        def fn(p):
            p.cables = new
        what = 'add the suggested cables' if doc.get('suggested') and not self.project.cables else \
            'set the suggested laid lengths'
        self.ctx.store.edit('cables', fn, f'Cables · {what}')
        self._preview = False
        self._say(f'Applied: {what}. Run again to check them.')

    def preview(self):
        self._preview = not self._preview
        self._invalidate()
        self._say('Preview: the cables drawn with the suggested laid lengths.' if self._preview else
                  'Preview off.')

    def explain_text(self) -> str:
        st = self.settings()
        return (
            'Each run between two attachment points crosses the moving joints between their links. With one revolute '
            'joint the cable is guided round the joint on a cylinder (the wrap radius: by default the nearer '
            'attachment\'s distance from the axis): the tangent from one point, the arc and the tangent to the other, '
            'unrolled with the height difference. The angle is not folded to half a turn: a joint that turns further '
            'winds the cable round it. With several joints the length is the straight chord.\n\n'
            'Each crossed joint is swept over its full range from the model file (an unlimited joint one turn round '
            'home) with the other joints at home, and every trajectory of the robot is played. The slack a joint '
            'needs is the largest minus the smallest length; the suggested laid length of a run is its largest '
            f'length plus a service loop ({st["loop_m"]:g} m; default 0.05 m).\n\n'
            'A pull is a run laid shorter than it needs, or a cable shorter than its runs together. The slack loop '
            'is drawn as a circular bow over the route, outwards from the joint; its smallest bend radius is '
            f'checked against the maker\'s minimum bend radius × {st["bend_factor"]:g} (the bend factor; default 1.5, '
            'a margin, not a maker\'s figure; the moving figure is used when the maker states one). A snag is the '
            'loop passing through one of the robot\'s links at home or along a trajectory (ray casting against the '
            'links\' colliding triangles, a seat of 20 mm round each attachment allowed).\n\n'
            'Known limits: no stiffness, weight or friction; the loop shape is an estimate; snags are tested at '
            'sampled poses and only against the cable\'s own robot; the run to the host is not included.')

    def explain(self):
        box = QMessageBox(self)
        box.setWindowTitle('How cable flex is computed')
        box.setText(self.explain_text())
        box.setModal(False)
        box.show()

    def export(self, path=None):
        if self.doc is None:
            return super().export(path)
        if path is None:
            path, _f = QFileDialog.getSaveFileName(self, 'Export the harness sheet', 'cables.json',
                                                   'JSON with provenance (*.json);;CSV (*.csv);;HTML (*.html)')
            if not path:
                return None
        ext = os.path.splitext(path)[1].lower()
        if ext not in ('.csv', '.html', '.htm'):
            return super().export(path)
        text = CS.to_csv(self.doc) if ext == '.csv' else CS.to_html(self.doc, self.project.name)
        try:
            with open(path, 'w', encoding='utf-8', newline='') as fh:
                fh.write(text)
        except OSError as e:
            self._say(f'Could not write {os.path.basename(path)}: {e.strerror or e}.')
            return None
        self._say(f'Exported the harness sheet to {os.path.basename(path)}')
        return path

    # ---------------------------------------------------------------- overlays
    def _invalidate(self):
        self._overlay_cache = None
        self._overlay_gen += 1
        self._refresh_views()

    def _refresh_views(self):
        self._redraw()
        win = self.ctx.window
        cv = getattr(win, 'camview', None) if win is not None else None
        if cv is not None and hasattr(cv, 'view'):
            try:
                cv.view.refresh_overlays()
            except Exception:                           # noqa: BLE001
                traceback.print_exc()

    def _redraw(self):
        win = self.ctx.window
        if win is not None and hasattr(win, 'viewport'):
            win.viewport.update()

    def polylines(self, wait: bool = False) -> list:
        """[(points (n, 3) world, status)] of every run at the joint state on screen, laid with its laid length (or
        the suggested one; in Preview, the suggested one).

        Laying the runs takes 100–150 ms on the bimanual and hand samples (measured, docs/performance.md), more than
        the UI thread may spend, so it runs on a worker thread: until it ends the views draw the last polylines
        computed, and they redraw when it does. wait=True computes them here (tests, exports)."""
        if self._overlay_cache is not None:
            return self._overlay_cache
        try:
            inputs = self._polyline_inputs()            # read on the UI thread: the project, the result, the joints
        except Exception:                               # noqa: BLE001 — an overlay must not take the page down
            traceback.print_exc()
            self._overlay_cache = []
            return []
        if wait:
            out = self._lay(inputs)
            self._overlay_cache = self._overlay_last = out
            return out
        if not self._overlay_running:
            self._overlay_running = True
            gen = self._overlay_gen
            sig = self._overlay_sig

            def work():
                out = self._lay(inputs)
                try:
                    sig.done.emit(gen, out)
                except RuntimeError:                    # the page was closed meanwhile
                    pass
            threading.Thread(target=work, daemon=True).start()
        return self._overlay_last

    def _polylines_ready(self, gen: int, out):
        self._overlay_running = False
        self._overlay_last = out
        if gen == self._overlay_gen:
            self._overlay_cache = out
        self._refresh_views()                           # a stale result: drawn, and the redraw starts a new one

    def _polyline_inputs(self) -> dict:
        doc = self.doc or {}
        return {'project': self.project, 'doc': doc, 'preview': self._preview, 'settings': self.settings(),
                'cables': self.cables() or [CB.clean_cable(c) for c in doc.get('suggested_cables') or []],
                'q': dict(self.ctx.store.q)}

    @staticmethod
    def _lay(inputs: dict) -> list:
        try:
            return CablesPage._compute_polylines(inputs)
        except Exception:                               # noqa: BLE001 — an overlay must not take the page down
            traceback.print_exc()
            return []

    @staticmethod
    def _compute_polylines(inputs: dict) -> list:
        from ...core.robot.model import get_model
        project, doc, st = inputs['project'], inputs['doc'], inputs['settings']
        res = {c['uid']: c for c in doc.get('cables') or []}
        out = []
        for c in inputs['cables']:
            atts = c['attachments']
            if len(atts) < 2 or len({a['robot_uid'] for a in atts}) != 1:
                continue
            ruid = atts[0]['robot_uid']
            if project.robot(ruid) is None:
                continue
            model = get_model(project, ruid)
            if any(a['link'] not in model.link_names for a in atts):
                continue
            q = inputs['q'].get(ruid)
            q = model.home if q is None or len(q) != model.nq else np.asarray(q, float)
            T, anc, ax = CB.posed(model, q[None])
            segs = {sg['index']: sg for sg in (res.get(c['uid']) or {}).get('segments') or []}
            for i in range(len(atts) - 1):
                run = CS._Run(model, atts[i], atts[i + 1], c['wrap_radius_m'])
                ev = run.evaluate(T, anc, ax, q[None])
                sg = segs.get(i) or {}
                s = sg.get('suggested_m') if inputs['preview'] else (atts[i]['length_to_next_m']
                                                                      or sg.get('suggested_m'))
                if s is None:
                    s = float(ev['L'][0]) + st['loop_m']
                P = run.loop(ev, float(s), int(st['points']))[0]
                out.append((P, sg.get('status', 'warning') if sg else 'warning'))
        return out

    def overlays(self, vp) -> list:
        from ..viewport3d import LineDraw
        if not self.show3d.isChecked():
            return []
        return [LineDraw(P.astype(np.float32), QColor(STATUS[sev]), width=3.0 if sev != 'good' else 2.0,
                         dash=self._preview) for P, sev in self.polylines()]

    def camview_overlays(self, cam_model, pose) -> list:
        """The cables seen through the Camera view, one entry per status."""
        from ...core import camrender as CR
        if not self.show3d.isChecked():
            return []
        by = {}
        for P, sev in self.polylines():
            by.setdefault(sev, []).extend(CR.visible_polylines(None, cam_model, pose, P, step=0.005))
        return [(polys, STATUS[sev], f'cables · {CS.WORDS[sev]}') for sev, polys in by.items() if polys]


register('cables', 'Cable flex across joints', CablesPage, menu='Analysis', icon='cable', workspace='placement',
         order=60)
