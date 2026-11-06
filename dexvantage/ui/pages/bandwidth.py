"""The Bandwidth, hosts and power page (spec §7.6): whether the host controllers, hubs and links carry every stream,
the recording budget per episode and per campaign, and USB bus power.

What it shows: the busiest controller, link or hub uplink as a share of its capacity (the headline, with its status
word; "tight" above the 80 % margin, a default), a margin bar per controller and per sensor link (ui/charts.py
conventions), the topology editor (hosts, their controllers, hubs, and which controller or hub each sensor plugs
into: every change is one undo step), the findings worst first, the disk budget and the bus power per device. In the
3D view each sensor is joined to its controller's hub point by a line in the controller's status colour.

Apply makes the study's suggestion (add a host with an assumed USB 3 controller when the rig has none, plug the
sensors that are not assigned into the controller with the most room, or move one sensor off a controller above the
margin), one undo step; Preview draws the move in the 3D view. The Camera view has no overlay for this study: a data
rate has no place in the image.
"""
from __future__ import annotations

import copy
import traceback

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QHeaderView, QLineEdit, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                             QWidget)

from ...core.studies import bandwidth as BW
from ..charts import ChartBase
from ..theme import STATUS, THEME, mono, sans
from ..viewport3d import LineDraw
from ..widgets import ScrubSpinBox, StatusLine, button, label
from . import register
from .base import StudyPage, row

KIND_TITLE = {'usb3': 'USB 3', 'usb2': 'USB 2.0', 'pcie': 'PCIe', 'gmsl': 'GMSL deserialiser'}
HOST_TYPE_KIND = {'usb_controller': 'usb3', 'compute': 'usb3', 'pcie_card': 'pcie', 'gmsl_deser': 'gmsl'}
UNKNOWN_COLOR = '#8f8e86'
NOT_STATED = 'not stated by the maker'


def _mbps(x) -> str:
    return '—' if x is None else f'{x:,.0f} Mbit/s' if x >= 100 else f'{x:,.1f} Mbit/s'


def _pct(x) -> str:
    return '—' if x is None else f'{x * 100:.0f} %'


def _col(status) -> QColor:
    return QColor(STATUS.get(status, UNKNOWN_COLOR))


class MarginChart(ChartBase):
    """One bar per controller, link or hub uplink: its load as a share of its capacity (to 120 %), the margin line
    (dashed) and 100 %; colour and word by status. Hover: the figures and where the capacity comes from."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = 'Load against capacity'
        self.subtitle = '% of capacity'
        self.rows: list = []            # (label, load or None, status, value text, [tip lines])
        self.warn = 0.8
        self.setMinimumHeight(48)

    def set_rows(self, rows, warn):
        self.rows, self.warn = list(rows), float(warn)
        self.hover, self.tip = None, None
        self.setMinimumHeight(26 + max(1, len(self.rows)) * 22 + 6)
        self.update()

    def _geom(self):
        return 22, 22, 210, 200

    def _x(self, frac):
        top, rh, x0, right = self._geom()
        bw = max(self.width() - x0 - right, 10)
        return x0 + min(max(frac, 0.0), 1.2) / 1.2 * bw

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.paint_title(p)
        top, rh, x0, right = self._geom()
        W = self.width()
        for i, (lab, load, status, val, _tip) in enumerate(self.rows):
            y = top + i * rh
            hot = self.hover == i
            if hot:
                p.fillRect(QRectF(0, y, W, rh), THEME.color('raised'))
            p.setFont(sans(11))
            p.setPen(THEME.color('ink2'))
            p.drawText(QRectF(4, y, x0 - 10, rh), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       p.fontMetrics().elidedText(lab, Qt.TextElideMode.ElideMiddle, int(x0 - 12)))
            if load is not None:
                r = QRectF(x0 + 1, y + 6, max(0.0, self._x(load) - x0 - 1), rh - 12)
                col = _col(status)
                p.setPen(QPen(THEME.color('ink'), 1) if hot else Qt.PenStyle.NoPen)
                p.setBrush(col.lighter(112) if hot else col)
                p.drawPath(self.bar_path(r, 3, 'right'))
            p.setFont(mono(11))
            p.setPen(THEME.color('ink'))
            p.drawText(QRectF(W - right + 6, y, right - 10, rh),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, val)
        if self.rows:
            y0, y1 = top + 2, top + len(self.rows) * rh - 2
            p.setPen(QPen(THEME.color('strong'), 1))
            p.drawLine(QPointF(x0 + 0.5, y0), QPointF(x0 + 0.5, y1))
            pen = QPen(THEME.color('muted'), 1, Qt.PenStyle.DashLine)
            p.setPen(pen)
            xw = self._x(self.warn)
            p.drawLine(QPointF(xw, y0), QPointF(xw, y1))
            p.setPen(QPen(THEME.color('ink2'), 1))
            x1 = self._x(1.0)
            p.drawLine(QPointF(x1, y0), QPointF(x1, y1))
            p.setFont(mono(10))
            p.setPen(THEME.color('muted'))
            p.drawText(QRectF(xw - 30, y1, 60, 14), Qt.AlignmentFlag.AlignCenter, f'{self.warn * 100:.0f} %')
            p.drawText(QRectF(x1 - 30, y1, 60, 14), Qt.AlignmentFlag.AlignCenter, '100 %')
        self.paint_tip(p)
        p.end()

    def mouseMoveEvent(self, e):
        top, rh, x0, _right = self._geom()
        i = int((e.position().y() - top) // rh)
        self.hover = i if 0 <= i < len(self.rows) and e.position().y() >= top else None
        if self.hover is not None:
            lab, load, status, val, tip = self.rows[self.hover]
            x = self._x(load) if load is not None else x0
            self.set_tip(QPointF(min(x, self.width() - 120), top + self.hover * rh + rh / 2), val, lab, *tip)
        else:
            self.tip = None
        self.update()


def _tree(headers) -> QTreeWidget:
    t = QTreeWidget()
    t.setHeaderLabels(headers)
    t.setUniformRowHeights(True)
    for k in range(1, len(headers)):
        t.header().setSectionResizeMode(k, QHeaderView.ResizeMode.ResizeToContents)
    t.setMinimumHeight(60)
    return t


def _fit(t: QTreeWidget, n=None):
    n = n if n is not None else t.topLevelItemCount()
    rh = max(t.sizeHintForRow(0) if t.topLevelItemCount() else 20, 22)
    t.setFixedHeight(t.header().sizeHint().height() + max(n, 1) * rh + 2 * t.frameWidth() + 18)


def _num_cell(it, k, text, tip=''):
    it.setText(k, text)
    it.setFont(k, mono(11))
    it.setTextAlignment(k, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    if tip:
        it.setToolTip(k, tip)


class BandwidthPage(StudyPage):
    """Bandwidth, hosts and power: each sensor's streams as the maker states them (width × height × fps × bits per
    pixel, plus a 5 % protocol overhead, a default) against its link and its controller, whose capacity is your
    measured figure or else the USB specification's effective bandwidth (an upper bound); the recording per episode
    and per campaign; and each bus-powered USB device's stated power against its port's current (USB 2.0: 500 mA,
    USB 3: 900 mA)."""
    STUDY = BW.STUDY_ID
    TITLE = 'Bandwidth, hosts and power'
    HEADLINE = 'The busiest controller, link or hub as a share of its capacity'

    # ---------------------------------------------------------------- body
    def build_body(self, layout):
        self._preview = None
        self._filling = False
        self.chart = MarginChart()
        layout.addWidget(self.chart)
        # topology editor
        layout.addWidget(label('Topology (each change is one undo step)', bold=True))
        self.c_host_entry = QComboBox()
        self.c_host_entry.setToolTip('A host from the library (its stated USB generation sets the controller\'s), or '
                                     'a host of your own')
        self.b_add_host = button('Add host', role='ghost', icon_name='plus', tip='Add the host with one controller')
        self.c_ctrl_kind = QComboBox()
        for k, t in KIND_TITLE.items():
            self.c_ctrl_kind.addItem(t, k)
        self.b_add_ctrl = button('Add controller', role='ghost', icon_name='plus',
                                 tip='Add a controller to the host selected in the tree (several ports can share '
                                     'one controller: lsusb -t shows which)')
        self.chk_powered = QCheckBox('Powered')
        self.chk_powered.setToolTip('A powered (self-powered) hub gives each port the high-power allowance; a '
                                    'bus-powered hub shares its own upstream port\'s current')
        self.b_add_hub = button('Add hub', role='ghost', icon_name='plus',
                                tip='Add a hub plugged into the controller or hub selected in the tree')
        self.b_remove = button('Remove', role='ghost', icon_name='trash',
                               tip='Remove the host, controller or hub selected in the tree (its sensors become '
                                   'not assigned)')
        layout.addWidget(row(self.c_host_entry, self.b_add_host, self.c_ctrl_kind, self.b_add_ctrl, self.chk_powered,
                             self.b_add_hub, self.b_remove))
        self.topo = _tree(['Host · controller · hub', 'Kind', 'Load', 'Capacity', 'Margin', 'Status'])
        self.topo.setRootIsDecorated(True)
        self.topo.setColumnWidth(0, 300)
        self.topo.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        layout.addWidget(self.topo)
        self.s_practical = ScrubSpinBox('', 'number', 0, unit='Mbit/s', width=130)
        self.s_practical.setRange(0.0, 1e6)
        self.s_practical.setSpecialValueText('—')              # 0: not stated
        self.s_practical.setToolTip('The selected controller\'s practical bandwidth, measured or from its manual (0: '
                                    'not stated: the USB specification\'s upper bound is used)')
        self.e_source = QLineEdit()
        self.e_source.setPlaceholderText('Source of the figure (a measurement, a manual)')
        self.b_set_practical = button('Set', role='ghost', tip='Store the figure and its source (one undo step)')
        layout.addWidget(row(label('Practical bandwidth', ink2=True), self.s_practical, self.e_source,
                             self.b_set_practical))
        layout.addWidget(label('Sensors and their links', bold=True))
        self.sens = _tree(['Sensor', 'Plugged into', 'Streams', 'Data rate', 'On the wire', 'Link load', 'Bus power'])
        self.sens.setRootIsDecorated(False)
        layout.addWidget(self.sens)
        self.stream_note = label('', muted=True, wrap=True, small=True)
        layout.addWidget(self.stream_note)
        layout.addWidget(label('Findings', bold=True))
        self.findings_w = QWidget()
        self.findings_l = QVBoxLayout(self.findings_w)
        self.findings_l.setContentsMargins(0, 0, 0, 0)
        self.findings_l.setSpacing(3)
        layout.addWidget(self.findings_w)
        # disk
        layout.addWidget(label('Recording', bold=True))
        self.s_episode = ScrubSpinBox('', 'number', 1, unit='s', width=90)
        self.s_episode.setRange(0.1, 86400)
        self.s_episode.setToolTip('Episode length (default 60 s)')
        self.s_episodes = ScrubSpinBox('', 'number', 0, width=90)
        self.s_episodes.setRange(1, 10_000_000)
        self.s_episodes.setToolTip('Episodes per campaign (default 100)')
        self.s_write = ScrubSpinBox('', 'number', 0, unit='MB/s', width=110)
        self.s_write.setRange(0.0, 1e6)
        self.s_write.setToolTip('The recorder\'s sustained write speed as its maker states it or as you measured it '
                                '(after any write cache is full); 0 = not stated')
        self.e_write_src = QLineEdit()
        self.e_write_src.setPlaceholderText('Source of the write speed')
        self.b_set_rec = button('Set', role='ghost', tip='Store the recording settings (one undo step)')
        layout.addWidget(row(label('Episode', ink2=True), self.s_episode, label('Episodes', ink2=True),
                             self.s_episodes, label('Sustained write', ink2=True), self.s_write, self.e_write_src,
                             self.b_set_rec))
        self.disk = label('', wrap=True)
        layout.addWidget(self.disk)
        self.assume = label('', muted=True, wrap=True, small=True)
        layout.addWidget(self.assume)
        # wiring
        self.b_add_host.clicked.connect(self.add_host)
        self.b_add_ctrl.clicked.connect(self.add_controller)
        self.b_add_hub.clicked.connect(self.add_hub)
        self.b_remove.clicked.connect(self.remove_selected)
        self.b_set_practical.clicked.connect(self._set_practical)
        self.b_set_rec.clicked.connect(self._set_recording)
        self.topo.currentItemChanged.connect(lambda *_: self._topo_selected())
        st = self.ctx.store
        st.changed.connect(lambda _s: self._fill_controls())
        st.project_replaced.connect(self._replaced)
        self._fill_controls()

    # ---------------------------------------------------------------- controls
    def _replaced(self):
        self.doc = None
        self._preview = None
        self._fill_controls()
        self.empty()

    def _rig(self) -> dict:
        return self.ctx.store.project.rig

    def _fill_controls(self):
        if self._filling:
            return
        self._filling = True
        try:
            from ...core import library as LB
            cur = self.c_host_entry.currentData()
            self.c_host_entry.clear()
            self.c_host_entry.addItem('Host of my own (no library entry)', '')
            for e in LB.entries('host'):
                self.c_host_entry.addItem(e.title, e.id)
            i = self.c_host_entry.findData(cur)
            self.c_host_entry.setCurrentIndex(max(i, 0))
            st = BW.settings_of(self.ctx.store.project)
            for w, v in ((self.s_episode, st['episode_s']), (self.s_episodes, st['episodes']),
                         (self.s_write, st['recorder']['write_mb_s'] or 0.0)):
                w.blockSignals(True)
                w.set_value(v)
                w.blockSignals(False)
            self.e_write_src.setText(st['recorder']['source'])
            self._fill_topology()
            self._fill_sensors()
        except Exception:                               # noqa: BLE001 — the controls must not take the page down
            traceback.print_exc()
        finally:
            self._filling = False

    def _result_rows(self):
        doc = self.doc or {}
        return ({r['uid']: r for r in doc.get('sensors') or []}, {c['uid']: c for c in doc.get('controllers') or []},
                {h['uid']: h for h in doc.get('hubs') or []})

    def _fill_topology(self):
        rig = self._rig()
        sel = self._selected_node()
        rows, crow, hrow = self._result_rows()
        names = {s.uid: s.name for s in self.ctx.store.project.variant.sensors}
        self.topo.clear()
        items = {}

        def figures(it, res):
            if res is None:
                return
            load = res.get('load_mbps')
            cap = res.get('capacity_mbps', res.get('uplink_capacity_mbps'))
            _num_cell(it, 2, _mbps(load))
            _num_cell(it, 3, _mbps(cap), res.get('capacity_source', ''))
            if 'margin_mbps' in res:
                _num_cell(it, 4, _mbps(res['margin_mbps']))
            stt = res.get('status', 'unknown')
            it.setText(5, BW.WORDS.get(stt, 'not checked') + (' · lower bound' if res.get('incomplete') else ''))
            it.setForeground(5, _col(stt))
        for h in rig.get('hosts') or []:
            hi = QTreeWidgetItem([h['name'], 'host'])
            hi.setData(0, Qt.ItemDataRole.UserRole, ('host', h['uid']))
            hi.setToolTip(0, f'Library entry: {h["library_id"]}' if h['library_id'] else 'No library entry')
            self.topo.addTopLevelItem(hi)
            items[h['uid']] = hi
            for c in h['controllers']:
                ci = QTreeWidgetItem([c['name'], KIND_TITLE.get(c['kind'], c['kind'])])
                ci.setData(0, Qt.ItemDataRole.UserRole, ('controller', c['uid']))
                figures(ci, crow.get(c['uid']))
                hi.addChild(ci)
                items[c['uid']] = ci
        pending = list(rig.get('hubs') or [])
        for _ in range(len(pending) + 1):               # hubs below hubs: place parents first
            left = []
            for b in pending:
                parent = items.get(b['upstream'])
                if parent is None:
                    left.append(b)
                    continue
                bi = QTreeWidgetItem([b['name'], 'powered hub' if b['powered'] else 'bus-powered hub'])
                bi.setData(0, Qt.ItemDataRole.UserRole, ('hub', b['uid']))
                figures(bi, hrow.get(b['uid']))
                parent.addChild(bi)
                items[b['uid']] = bi
            pending = left
        for b in pending:                               # a hub whose upstream is gone
            bi = QTreeWidgetItem([b['name'], 'hub · plugged into nothing'])
            bi.setData(0, Qt.ItemDataRole.UserRole, ('hub', b['uid']))
            self.topo.addTopLevelItem(bi)
            items[b['uid']] = bi
        for k in rig.get('links') or []:
            parent = items.get(k['via'])
            if parent is None or k['sensor_uid'] not in names:
                continue
            si = QTreeWidgetItem([names[k['sensor_uid']], 'sensor'])
            si.setData(0, Qt.ItemDataRole.UserRole, ('sensor', k['sensor_uid']))
            r = rows.get(k['sensor_uid'])
            if r is not None:
                _num_cell(si, 2, _mbps(r['wire_mbps']))
                _num_cell(si, 3, _mbps(r['link_capacity_mbps']), r['link_capacity_source'])
                _num_cell(si, 4, _mbps(r['link_margin_mbps']))
                si.setText(5, BW.WORDS.get(r['link_status'], 'not checked'))
                si.setForeground(5, _col(r['link_status']))
            parent.addChild(si)
        self.topo.expandAll()
        n = sum(1 for _ in self._walk(self.topo.invisibleRootItem()))
        _fit(self.topo, n)
        if sel is not None:
            for it in self._walk(self.topo.invisibleRootItem()):
                if it.data(0, Qt.ItemDataRole.UserRole) == sel:
                    self.topo.setCurrentItem(it)
                    break

    def _walk(self, it):
        for k in range(it.childCount()):
            c = it.child(k)
            yield c
            yield from self._walk(c)

    def _selected_node(self):
        it = self.topo.currentItem() if hasattr(self, 'topo') else None
        return None if it is None else it.data(0, Qt.ItemDataRole.UserRole)

    def _topo_selected(self):
        sel = self._selected_node()
        c = self._controller(sel[1]) if sel and sel[0] == 'controller' else None
        self.s_practical.setEnabled(c is not None)
        self.e_source.setEnabled(c is not None)
        self.b_set_practical.setEnabled(c is not None)
        if c is not None:
            self.s_practical.set_value(c.get('practical_mbps') or 0.0)
            self.e_source.setText(c.get('source') or '')

    def _controller(self, uid):
        for h in self._rig().get('hosts') or []:
            for c in h['controllers']:
                if c['uid'] == uid:
                    return c
        return None

    def _targets(self) -> list:
        """[(uid, label)] of every controller and hub a sensor can plug into."""
        rig = self._rig()
        out = []
        for h in rig.get('hosts') or []:
            for c in h['controllers']:
                out.append((c['uid'], f'{h["name"]} · {c["name"]}'))
        for b in rig.get('hubs') or []:
            out.append((b['uid'], BW.hub_label(b['name'])))
        return out

    def _fill_sensors(self):
        p = self.ctx.store.project
        rows, _c, _h = self._result_rows()
        links = {k['sensor_uid']: k for k in p.rig.get('links') or []}
        targets = self._targets()
        self.sens.clear()
        for s in p.variant.sensors:
            r = rows.get(s.uid)
            it = QTreeWidgetItem([s.name, ''])
            self.sens.addTopLevelItem(it)
            combo = QComboBox()
            combo.addItem('— not assigned', '')
            for uid, lab in targets:
                combo.addItem(lab, uid)
            via = (links.get(s.uid) or {}).get('via', '')
            i = combo.findData(via)
            combo.setCurrentIndex(i if i >= 0 else 0)
            combo.setToolTip('The controller or hub this sensor\'s cable plugs into (one undo step)')
            combo.activated.connect(lambda _i, uid=s.uid, cb=combo: self.plug(uid, cb.currentData() or ''))
            self.sens.setItemWidget(it, 1, combo)
            if r is None:
                it.setText(2, 'switched off' if not s.enabled else '')
                continue
            st = ', '.join(x['label'] for x in r['streams']) or '—'
            it.setText(2, ('default: ' if r['default_streams'] else '') + st)
            it.setToolTip(2, ('The default streams (the depth and RGB streams at the chosen size, the rate nearest '
                              '30 fps, a default). ' if r['default_streams'] else '') +
                          '; '.join(f'{x["label"]}: {_mbps(x["payload_mbps"])}' for x in r['streams']))
            _num_cell(it, 3, _mbps(r['payload_mbps']), r['rate_source'] if r['payload_mbps'] is not None else
                      f'{NOT_STATED}: ' + (', '.join(r['not_stated']) or r['rate_source']))
            _num_cell(it, 4, _mbps(r['wire_mbps']), f'+ {r["overhead"] * 100:g} % overhead: {r["overhead_source"]}')
            _num_cell(it, 5, _pct(r['link_load']), r['link_capacity_source'])
            it.setForeground(5, _col(r['link_status']))
            pw = '—' if r['draw_ma'] is None else f'{r["draw_ma"]:,.0f}' + (
                f' / {r["allowance_ma"]:,.0f} mA' if r['allowance_ma'] else ' mA')
            _num_cell(it, 6, pw, r['power_note'] or (NOT_STATED if r['power_w'] is None else ''))
            it.setForeground(6, _col(r['power_status']))
        _fit(self.sens)

    # ---------------------------------------------------------------- edits (each one undo step)
    def _edit_rig(self, fn, text):
        from ...core.project import rig_from_dict

        def wrap(p):
            rig = copy.deepcopy(p.rig)
            fn(rig)
            p.rig = rig_from_dict(rig)
        self.ctx.store.edit('rig', wrap, text)

    def plug(self, sensor_uid: str, via: str):
        """Plug a sensor into a controller or hub ('' takes it out)."""
        p = self.ctx.store.project
        s = p.variant.sensor(sensor_uid)
        if s is None:
            return
        cur = next((k for k in p.rig.get('links') or [] if k['sensor_uid'] == sensor_uid), None)
        if (cur or {}).get('via', '') == via:
            return
        name = dict(self._targets()).get(via, '')

        def fn(rig):
            k = next((x for x in rig['links'] if x['sensor_uid'] == sensor_uid), None)
            if not via:
                rig['links'] = [x for x in rig['links'] if x['sensor_uid'] != sensor_uid]
            elif k is None:
                rig['links'].append({'sensor_uid': sensor_uid, 'stream_ids': [], 'via': via, 'cable_uid': ''})
            else:
                k['via'] = via
        self._edit_rig(fn, f'Plug {s.name} into {name}' if via else f'Unplug {s.name}')

    def add_host(self, library_id=None):
        from ...core import library as LB
        from ...core.project import new_uid
        lid = self.c_host_entry.currentData() if library_id is None else library_id
        e = LB.get(lid) if lid else None
        kind = HOST_TYPE_KIND.get(((e.data.get('host') or {}).get('type') if e is not None else None), 'usb3')
        n = len(self._rig().get('hosts') or []) + 1
        name = e.title if e is not None else f'Host {n}'

        def fn(rig):
            rig['hosts'].append({'uid': new_uid(), 'name': name, 'library_id': lid or '',
                                 'controllers': [{'uid': new_uid(), 'name': f'{KIND_TITLE[kind]} controller 1',
                                                  'kind': kind, 'practical_mbps': None, 'source': ''}]})
        self._edit_rig(fn, f'Add host {name}')

    def _host_of_selection(self):
        sel = self._selected_node()
        rig = self._rig()
        if sel is None:
            hs = rig.get('hosts') or []
            return hs[-1]['uid'] if hs else None
        if sel[0] == 'host':
            return sel[1]
        for h in rig.get('hosts') or []:
            if any(c['uid'] == sel[1] for c in h['controllers']):
                return h['uid']
        return None

    def add_controller(self, kind=None):
        from ...core.project import new_uid
        hu = self._host_of_selection()
        if hu is None:
            if self.ctx.window is not None:
                self.ctx.window.msg('Add a host first, then a controller to it.', 5000)
            return
        kind = kind or self.c_ctrl_kind.currentData()

        def fn(rig):
            h = next(x for x in rig['hosts'] if x['uid'] == hu)
            h['controllers'].append({'uid': new_uid(), 'name': f'{KIND_TITLE[kind]} controller '
                                                                f'{len(h["controllers"]) + 1}',
                                     'kind': kind, 'practical_mbps': None, 'source': ''})
        self._edit_rig(fn, f'Add a {KIND_TITLE[kind]} controller')

    def add_hub(self, upstream=None, powered=None):
        from ...core.project import new_uid
        sel = self._selected_node()
        up = upstream or (sel[1] if sel and sel[0] in ('controller', 'hub') else None)
        if up is None:
            if self.ctx.window is not None:
                self.ctx.window.msg('Select the controller or hub the new hub plugs into.', 5000)
            return
        pw = self.chk_powered.isChecked() if powered is None else bool(powered)
        n = len(self._rig().get('hubs') or []) + 1

        def fn(rig):
            rig['hubs'].append({'uid': new_uid(), 'name': f'Hub {n}', 'powered': pw, 'upstream': up})
        self._edit_rig(fn, f'Add {"a powered" if pw else "a bus-powered"} hub')

    def remove_selected(self):
        sel = self._selected_node()
        if sel is None or sel[0] == 'sensor':
            if sel is not None:
                self.plug(sel[1], '')
            return
        kind, uid = sel

        def fn(rig):
            gone = {uid}
            if kind == 'host':
                h = next((x for x in rig['hosts'] if x['uid'] == uid), None)
                gone |= {c['uid'] for c in (h or {}).get('controllers', [])}
                rig['hosts'] = [x for x in rig['hosts'] if x['uid'] != uid]
            elif kind == 'controller':
                for h in rig['hosts']:
                    h['controllers'] = [c for c in h['controllers'] if c['uid'] != uid]
            changed = True
            while changed:                              # hubs below what goes go too
                changed = False
                for b in list(rig['hubs']):
                    if b['upstream'] in gone or b['uid'] in gone:
                        gone.add(b['uid'])
                        rig['hubs'].remove(b)
                        changed = True
            rig['links'] = [k for k in rig['links'] if k['via'] not in gone]
        self._edit_rig(fn, f'Remove the {kind}')

    def _set_practical(self):
        sel = self._selected_node()
        if not sel or sel[0] != 'controller':
            return
        v = float(self.s_practical.value()) or None
        src = self.e_source.text().strip()

        def fn(rig):
            for h in rig['hosts']:
                for c in h['controllers']:
                    if c['uid'] == sel[1]:
                        c['practical_mbps'], c['source'] = v, src
        self._edit_rig(fn, 'Controller bandwidth: ' + (f'{v:g} Mbit/s' if v else 'not stated'))

    def _set_recording(self):
        ep, n = float(self.s_episode.value()), int(self.s_episodes.value())
        w = float(self.s_write.value()) or None
        src = self.e_write_src.text().strip()

        def fn(p):
            st = copy.deepcopy(p.analysis.studies.get(BW.STUDY_ID) or {})
            st['episode_s'], st['episodes'] = ep, n
            rec = dict(st.get('recorder') or {})
            rec['write_mb_s'], rec['source'] = w, src
            st['recorder'] = rec
            p.analysis.studies[BW.STUDY_ID] = st
        self.ctx.store.edit('analysis', fn, f'Recording: {ep:g} s episodes, {n} per campaign')

    # ---------------------------------------------------------------- result
    def empty(self):
        self.head.set('—', '', 'warning', 'not run yet', 'Run the bandwidth study to see whether the hosts carry every '
                                                         'stream.')
        self.chart.set_rows([], 0.8)
        self.suggestion_ready(False)

    def show_result(self, doc: dict):
        self._preview = None
        hl = doc.get('headline') or {}
        st = doc.get('settings') or {}
        warn = st.get('margin_warn', 0.8)
        sug = doc.get('suggestion')
        bad = [f for f in doc.get('findings') or [] if f['severity'] in ('critical', 'serious', 'warning')]
        act = sug['text'] if sug else (bad[0]['message'] if bad else 'Nothing to change: every figure is within its '
                                                                    'budget.')
        tip = (f'Status: "tight" above {warn * 100:g} % of a capacity (a default, setting margin_warn), "over budget" '
               'above 100 %. Capacities are your measured figures, else the USB specification\'s effective bandwidth '
               '(an upper bound).')
        if doc.get('message'):
            act = doc['message']
        if hl.get('busiest_load_pct') is None:
            self.head.set('—', '', hl.get('status', 'warning'), hl.get('status_word', ''), act, tip)
        else:
            self.head.set(f'{hl["busiest_load_pct"]:.0f}', '%', hl['status'], hl['status_word'],
                          f'Busiest: {hl["busiest"]}. {act}', tip)
        self.suggestion_ready(bool(sug))
        rows = []
        for c in doc.get('controllers') or []:
            rows.append((c['label'], c['load'], c['status'],
                         f'{_pct(c["load"])} · {_mbps(c["margin_mbps"])} left' if c['load'] is not None else
                         'not checked', [f'{_mbps(c["load_mbps"])} of {_mbps(c["capacity_mbps"])}',
                                         c['capacity_source']]))
        for h in doc.get('hubs') or []:
            if h['load'] is not None:
                rows.append((f'{BW.hub_label(h["name"])} uplink', h['load'], h['status'], _pct(h['load']),
                             [f'{_mbps(h["load_mbps"])} of {_mbps(h["uplink_capacity_mbps"])}']))
        for r in doc.get('sensors') or []:
            if r['linked']:
                rows.append((f'{r["name"]} link', r['link_load'], r['link_status'],
                             _pct(r['link_load']) if r['link_load'] is not None else 'not checked',
                             [f'{_mbps(r["wire_mbps"])} of {_mbps(r["link_capacity_mbps"])}',
                              r['link_capacity_source']]))
        self.chart.set_rows(rows, warn)
        while self.findings_l.count():
            w = self.findings_l.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        for f in doc.get('findings') or []:
            sev = f['severity'] if f['severity'] in STATUS else ''
            self.findings_l.addWidget(StatusLine(sev, f['message'], wrap=True))
        d = doc.get('disk') or {}
        self.disk.setText(
            f'Recorded {_mbps(d.get("recorded_mbps"))} ({d.get("recorded_mb_s", 0):,.1f} MB/s, '
            f'{d.get("gb_per_hour", 0):,.0f} GB per hour): {d.get("gb_per_episode", 0):,.2f} GB per '
            f'{d.get("episode_s", 0):g} s episode, {d.get("tb_per_campaign", 0):,.3f} TB per campaign of '
            f'{d.get("episodes", 0):,} episodes. Sustained write needed {d.get("write_needed_mb_s", 0):,.1f} MB/s '
            f'(× {st.get("write_headroom", 1.3):g} headroom, a default); the recorder: '
            + (f'{d["write_mb_s"]:,.0f} MB/s ({d.get("write_source") or "no source given"}).'
               if d.get('write_mb_s') else 'not stated.')
            + (f' {len(d["incomplete"])} sensor(s) without a stated rate are not counted.' if d.get('incomplete')
               else ''))
        self.assume.setText('\n'.join(list(doc.get('notes') or []) + list(doc.get('assumptions') or [])
                                      + ['References: ' + '; '.join(doc.get('references') or [])]))
        notes = []
        for r in doc.get('sensors') or []:
            notes += [f'{r["name"]}: {p}' for p in r.get('stream_problems') or []]
        self.stream_note.setText('\n'.join(notes))
        self._fill_controls()
        if self.ctx.window is not None:
            self.ctx.window.viewport.update()

    # ---------------------------------------------------------------- Apply / Preview
    def apply(self):
        sug = (self.doc or {}).get('suggestion')
        if not sug:
            return

        def fn(p):
            BW.apply_suggestion(p, sug)
        self.ctx.store.edit('rig', fn, 'Bandwidth: ' + sug['text'])
        self.run()

    def preview(self):
        sug = (self.doc or {}).get('suggestion')
        self._preview = sug if sug and self._preview is None else None
        if self.ctx.window is not None:
            self.ctx.window.msg(('Preview: ' + sug['text'] + ' (dashed lines in the 3D view)') if self._preview else
                                'Preview off', 8000)
            self.ctx.window.viewport.update()

    # ---------------------------------------------------------------- 3D: sensors joined per controller
    def overlays(self, vp) -> list:
        doc = self.doc
        if not doc:
            return []
        from ...core import scenegeom as SG
        p = self.ctx.store.project
        pos = {}
        for s in p.variant.sensors:
            try:
                pos[s.uid] = np.asarray(SG.sensor_world_pose(p, s, self.ctx.store.q).position, float)
            except Exception:                           # noqa: BLE001 — a sensor that cannot be posed is not drawn
                continue
        items = []
        groups = {}
        for r in doc.get('sensors') or []:
            if r.get('controller_uid') and r['uid'] in pos:
                groups.setdefault(r['controller_uid'], []).append(r['uid'])
        cstat = {c['uid']: c['status'] for c in doc.get('controllers') or []}
        for cu, uids in groups.items():
            P = np.array([pos[u] for u in uids])
            hub = P.mean(0) + np.array([0.0, 0.0, 0.08])
            segs = np.stack([P, np.repeat(hub[None], len(P), 0)], 1)
            items.append(LineDraw(segs, _col(cstat.get(cu, 'unknown')), 2.0, segments=True, depth=False))
        sug = self._preview
        if sug:
            for m in sug.get('moves') or []:
                a = pos.get(m['sensor_uid'])
                if a is None:
                    continue
                dest = [pos[u] for u in groups.get(m['to'], []) if u in pos and u != m['sensor_uid']]
                b = (np.mean(dest, 0) if dest else a) + np.array([0.0, 0.0, 0.08])
                if np.allclose(a, b):
                    b = a + np.array([0.0, 0.0, 0.15])
                items.append(LineDraw(np.stack([a, b])[None], THEME.color('ink', 0.9), 1.6, dash=True,
                                      segments=True, depth=False, dash_len=0.01))
        return items

    def camview_overlays(self, cam_model, pose) -> list:
        """None: a data rate has no place in the image (module docstring)."""
        return []


register('bandwidth', 'Bandwidth, hosts and power', BandwidthPage, menu='Analysis', icon='cable', action_id=None,
         workspace=None, order=60)
