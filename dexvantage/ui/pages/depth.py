"""The Depth validity page (spec §7.2): where each depth sensor gives a depth reading in the task volumes, why it does
not elsewhere, and how noisy the depth is expected to be against the maker's figure.

What it shows: the share of the task volumes that at least one depth sensor covers with valid depth (the headline,
with a status word; thresholds 80 % and 50 % are defaults), the breakdown of the selected sensor's volume by the
first condition that fails (a bar chart in the ui/charts.py conventions), a table of the expected depth error
σ_z = z² σ_d / (f B) beside the maker's stated RMS figure, the fingertips' depth against min-Z, the material
warnings, and in the 3D view the invalid voxels of the selected sensor coloured by reason. Every depth sensor's min-Z
blind zone, the part of its field of view nearer than min-Z, is drawn as a translucent volume from the sensor's
pose at the joint state on screen, on every page (View ▸ Depth min-Z zones), so a wrist camera's zone moves with the
wrist.

The depth resolution is a per-sensor setting (one undo step when changed; the default is the device's native depth
stream, labelled as the default). Run passes the 3D view's joint state to the study (not stored in the project).
Apply makes the suggested change (a resolution with a smaller min-Z, when that gains depth-valid volume).
"""
from __future__ import annotations

import copy
import traceback

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QCheckBox, QComboBox, QHeaderView, QTreeWidget, QTreeWidgetItem

from ...core.studies import depth_validity as DV
from ..charts import ChartBase
from ..theme import THEME, mono, sans
from ..viewport3d import LineDraw, TriDraw
from ..widgets import SegmentedControl, StatusLine, label
from . import register
from .base import StudyPage, row

NOT_STATED = 'not stated by the maker'
# what the maker's depth-error figure measures (library rms_error.metric), in the 'Maker states' column
METRIC_WORDS = {'rms': 'RMS', 'median_abs': 'median abs.', 'std_dev': 'std. dev.', 'unspecified': '(metric not stated)'}
METRIC_TIPS = {'median_abs': 'a median absolute error, not an RMS.', 'std_dev': 'a standard deviation in length units.',
               'unspecified': 'an "accuracy" whose metric the maker does not define.'}
ZONE_ALPHA = 0.16
# categories, always shown with their word (these are not status colours)
REASON_COLOR = {'valid': '#3987e5', 'fov': '#8f8e86', 'min_z': '#d6457a', 'max_range': '#8a63d2',
                'occluded': '#e0883a', 'projector': '#c9a227', 'material': '#2fa3a8'}
SHOW_DEFAULT = {'fov': False, 'min_z': True, 'max_range': True, 'occluded': True, 'projector': True,
                'material': True}
OVERLAY_DRAW_MAX = 1500          # crosses drawn per reason in the 3D view
ADVICE = {
    'fov': 'aim or move the sensor so the task volume fills its view (for stereo, both imagers must see a point)',
    'min_z': 'move the sensor back, or choose a depth resolution with a smaller min-Z',
    'max_range': 'move the sensor closer, or choose a device with a longer range',
    'occluded': 'raise the sensor or move it off the gripper\'s axis so the fingers and the arm shadow less',
    'projector': 'the projector does not light that part: aim the sensor at it, or turn the projector need off for '
                 'well-textured scenes',
    'material': 'those surfaces may still give no depth: test them, or plan for a matte coating or another sensor',
}


def _mm(x, nd=0) -> str:
    return '—' if x is None else f'{x * 1000:.{nd}f} mm'


def _pct(x, nd=1) -> str:
    return '—' if x is None else f'{x:.{nd}f} %'


class ConditionChart(ChartBase):
    """Horizontal bars, one per condition: the share of the task volume (%) whose first failing condition it is, and
    the depth-valid share; hover gives the value and what it means."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = 'Why the task volume is not depth-valid'
        self.subtitle = '% of the task volume'
        self.rows: list = []           # (key, title, pct, text)
        self.setMinimumHeight(22 + 7 * 22 + 4)

    def set_rows(self, rows):
        self.rows = list(rows)
        self.hover = None
        self.tip = None
        self.setMinimumHeight(22 + max(1, len(self.rows)) * 22 + 4)
        self.update()

    def _geom(self):
        return 22, 22, 190

    def _bar(self, i, pct):
        top, rh, x0 = self._geom()
        bw = max(self.width() - x0 - 70, 10)
        return QRectF(x0 + 1, top + i * rh + 6, max(0.0, min(pct, 100.0) / 100.0 * bw), rh - 12)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.paint_title(p)
        top, rh, x0 = self._geom()
        W = self.width()
        for i, (key, title, pct, _text) in enumerate(self.rows):
            y = top + i * rh
            hot = self.hover == i
            if hot:
                p.fillRect(QRectF(0, y, W, rh), THEME.color('raised'))
            p.setFont(sans(11))
            p.setPen(THEME.color('ink2'))
            p.drawText(QRectF(4, y, x0 - 10, rh), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, title)
            r = self._bar(i, pct)
            if r.width() > 0:
                col = QColor(REASON_COLOR.get(key, '#888888'))
                p.setPen(QPen(THEME.color('ink'), 1) if hot else Qt.PenStyle.NoPen)
                p.setBrush(col.lighter(112) if hot else col)
                p.drawPath(self.bar_path(r, 3, 'right'))
            p.setFont(mono(12))
            p.setPen(THEME.color('ink'))
            p.drawText(QRectF(W - 70, y, 64, rh), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       f'{pct:.1f} %')
        if self.rows:
            p.setPen(QPen(THEME.color('strong'), 1))
            p.drawLine(QPointF(x0 + 0.5, top + 3), QPointF(x0 + 0.5, top + len(self.rows) * rh - 3))
        self.paint_tip(p)
        p.end()

    def mouseMoveEvent(self, e):
        top, rh, _x0 = self._geom()
        i = int((e.position().y() - top) // rh)
        self.hover = i if 0 <= i < len(self.rows) and e.position().y() >= top else None
        if self.hover is not None:
            key, title, pct, text = self.rows[self.hover]
            r = self._bar(self.hover, pct)
            self.set_tip(QPointF(min(r.right(), self.width() - 90), r.center().y()), f'{pct:.2f} % of the task volume',
                         title, text)
        else:
            self.tip = None
        self.update()


def _tree(headers) -> QTreeWidget:
    t = QTreeWidget()
    t.setRootIsDecorated(False)
    t.setHeaderLabels(headers)
    t.setUniformRowHeights(True)
    for k in range(1, len(headers)):
        t.header().setSectionResizeMode(k, QHeaderView.ResizeMode.ResizeToContents)
    t.setMinimumHeight(60)
    return t


def _fit(t: QTreeWidget):
    """Size a table to its rows (no empty band under them)."""
    n = t.topLevelItemCount()
    rh = t.sizeHintForRow(0) if n else 20
    t.setFixedHeight(t.header().sizeHint().height() + max(n, 1) * max(rh, 18) + 2 * t.frameWidth() + 18)


class DepthPage(StudyPage):
    """Depth validity: the share of the task volumes (voxels weighted by their volume) where a depth sensor gives a
    depth reading at the pose on screen: inside both stereo imagers' fields of view, beyond min-Z for the chosen
    resolution and within the maximum range, seen by both imagers, inside the projector cone when the sensor needs
    its texture, and not on a surface tagged transparent, IR-absorbing or specular. σ_z = z² σ_d / (f B) is the
    expected depth error."""
    STUDY = DV.STUDY_ID
    TITLE = 'Depth validity'
    HEADLINE = 'Task volume with valid depth from at least one depth sensor'

    # ---------------------------------------------------------------- body
    def build_body(self, layout):
        self._zone_cache: dict = {}
        self._overlay_key = None
        self._overlay_items: list = []
        self._preview = None
        self.sel_uid = ''
        self.c_sensor = QComboBox()
        self.c_sensor.setToolTip('The depth sensor the chart, the notes and the 3D voxels are for')
        self.c_res = QComboBox()
        self.c_res.setToolTip('The depth stream resolution: min-Z and the focal length in pixels depend on it. '
                              'The default is the device\'s native depth stream.')
        self.pose = SegmentedControl([('current', 'Pose on screen'), ('home', 'Home pose')], 'current')
        self.pose.setToolTip('Run the study at the joint state on screen (finger shadows move with the arm) or at '
                             'the robots\' home pose')
        layout.addWidget(row(label('Sensor', ink2=True), self.c_sensor, label('Resolution', ink2=True), self.c_res,
                             self.pose))
        self.res_note = label('', muted=True, wrap=True)
        layout.addWidget(self.res_note)
        self.chart = ConditionChart()
        layout.addWidget(self.chart)
        self.occl_note = label('', muted=True, wrap=True)
        layout.addWidget(self.occl_note)
        self.fingers = StatusLine('', '', wrap=True)
        layout.addWidget(self.fingers)
        self.mats = label('', wrap=True)
        layout.addWidget(self.mats)
        layout.addWidget(label('Expected depth error against the maker\'s figure', bold=True))
        self.noise = _tree(['Sensor', 'Resolution', 'f', 'Baseline', 'σ_d', 'σ_z median [5–95 %]', 'σ_z / z',
                            'Model at the maker\'s distance', 'Maker states', 'σ_d matching the maker'])
        layout.addWidget(self.noise)
        self.noise_note = label('', muted=True, wrap=True, small=True)
        layout.addWidget(self.noise_note)
        layout.addWidget(label('Depth sensors', bold=True))
        self.sens = _tree(['Sensor', 'Device', 'Depth-valid', 'min-Z', 'Max range', 'Projector', 'Hidden from one',
                           'Hidden from all'])
        layout.addWidget(self.sens)
        layout.addWidget(label('Invalid voxels in the 3D view (while this page is shown)', bold=True))
        self.show = {}
        boxes = []
        for k in DV.REASONS:
            cb = QCheckBox(DV.REASON_TITLE[k])
            cb.setChecked(SHOW_DEFAULT[k])
            cb.setStyleSheet(f'QCheckBox {{ color: {REASON_COLOR[k]}; }}')
            cb.setToolTip(DV.REASON_TEXT[k])
            cb.toggled.connect(lambda _on: self._redraw())
            self.show[k] = cb
            boxes.append(cb)
        layout.addWidget(row(*boxes[:3]))
        layout.addWidget(row(*boxes[3:]))
        self.assume = label('', muted=True, wrap=True, small=True)
        layout.addWidget(self.assume)
        self.c_sensor.currentIndexChanged.connect(self._sensor_changed)
        self.c_res.activated.connect(self._res_chosen)
        st = self.ctx.store
        st.changed.connect(lambda _s: self._fill_controls())
        st.project_replaced.connect(self._replaced)
        self.ctx.overlay_sources.append(self.zones)
        self._fill_controls()

    # ---------------------------------------------------------------- controls
    def _sensors(self) -> list:
        try:
            return DV.depth_sensors(self.ctx.store.project)
        except Exception:                               # noqa: BLE001 — the controls must not take the page down
            traceback.print_exc()
            return []

    def _replaced(self):
        self.doc = None
        self._zone_cache.clear()
        self._fill_controls()
        self.empty()

    def _fill_controls(self):
        ss = self._sensors()
        self.c_sensor.blockSignals(True)
        self.c_sensor.clear()
        for s in ss:
            self.c_sensor.addItem(s.name, s.uid)
        uids = [s.uid for s in ss]
        if self.sel_uid not in uids:
            self.sel_uid = uids[0] if uids else ''
        if self.sel_uid:
            self.c_sensor.setCurrentIndex(uids.index(self.sel_uid))
        self.c_sensor.blockSignals(False)
        self._fill_res()

    def _sensor(self):
        return next((s for s in self._sensors() if s.uid == self.sel_uid), None)

    def _fill_res(self):
        self.c_res.blockSignals(True)
        self.c_res.clear()
        s = self._sensor()
        note = ''
        if s is None:
            self.c_res.setEnabled(False)
            note = 'No depth sensor in this rig: add one (Sensors) and choose its device from the library.'
        else:
            try:
                ds = DV.setup(self.ctx.store.project, s)
            except DV.DepthSetupError as e:
                ds = None
                note = str(e)
            self.c_res.setEnabled(ds is not None)
            if ds is not None:
                native, _why = DV.native_depth_mode(ds.entry)
                for w, h, m in DV.resolution_options(ds.entry):
                    mz = f'min-Z {m * 1000:.0f} mm' if m is not None else 'min-Z —'
                    tag = ' (default)' if (w, h) == native else ''
                    self.c_res.addItem(f'{w} × {h} · {mz}{tag}', [w, h])
                    if (w, h) == tuple(ds.resolution):
                        self.c_res.setCurrentIndex(self.c_res.count() - 1)
                note = (f'{ds.resolution_note}. min-Z {_mm(ds.min_z_m)} ({ds.min_z_note}); maximum range '
                        f'{"—" if ds.max_range_m is None else f"{ds.max_range_m:g} m"} ({ds.max_range_note}). '
                        f'Projector: {ds.projector_note}.')
        self.c_res.blockSignals(False)
        self.res_note.setText(note)

    def _sensor_changed(self, i):
        self.sel_uid = self.c_sensor.itemData(i) or ''
        self._fill_res()
        if self.doc is not None:
            self.show_result(self.doc)
        self._redraw()

    def set_resolution(self, uid: str, res) -> None:
        """Set a sensor's depth resolution (one undo step); None goes back to the default."""
        s = next((x for x in self._sensors() if x.uid == uid), None)
        if s is None:
            return

        def fn(p):
            st = copy.deepcopy(p.analysis.studies.get(DV.STUDY_ID) or {})
            sens = st.setdefault('sensors', {})
            cfg = sens.setdefault(uid, {})
            if res is None:
                cfg.pop('resolution', None)
            else:
                cfg['resolution'] = [int(res[0]), int(res[1])]
            if not cfg:
                sens.pop(uid, None)
            p.analysis.studies[DV.STUDY_ID] = st
        text = (f'Depth resolution of {s.name}: ' + ('the default' if res is None else f'{res[0]} × {res[1]}'))
        self.ctx.store.edit('analysis', fn, text)
        self._zone_cache.clear()

    def _res_chosen(self, i):
        res = self.c_res.itemData(i)
        if res is None or not self.sel_uid:
            return
        try:
            cur = DV.setup(self.ctx.store.project, self._sensor()).resolution
        except DV.DepthSetupError:
            return
        if tuple(res) != tuple(cur):
            self.set_resolution(self.sel_uid, res)
            self.run()

    # ---------------------------------------------------------------- run
    def live_q(self) -> dict:
        return {uid: [float(x) for x in np.asarray(q, float).reshape(-1)] for uid, q in self.ctx.store.q.items()}

    def run(self):
        """Run at the pose on screen (passed with the run, not stored) or at home."""
        if self.pose.value() == 'home':
            self.ctx.analysis.run(self.STUDY, {'pose': 'home', 'q': {}})
        else:
            self.ctx.analysis.run(self.STUDY, {'pose': None, 'q': self.live_q()})

    def _failed(self, sid, message):
        if sid != self.STUDY:
            return
        from ..store import failure_text
        first, _explained = failure_text(message, 'Depth validity')       # the window logs the traceback
        self.head.set('—', '', 'critical', 'could not run', first)

    # ---------------------------------------------------------------- result
    def empty(self):
        self.head.set('—', '', 'warning', 'not run yet', 'Run depth validity to see where the depth sensors give '
                                                         'valid depth in the task volumes.')
        self.chart.set_rows([])

    def _sel(self, doc):
        ss = doc.get('sensors') or []
        return next((s for s in ss if s['uid'] == self.sel_uid), ss[0] if ss else None)

    def show_result(self, doc: dict):
        self._preview = None
        ss = doc.get('sensors') or []
        rig = (doc.get('rig') or {}).get('valid_pct')
        if not ss:
            self.head.set('—', '', 'warning', 'nothing to show', doc.get('message') or 'No depth sensor analysed.')
            self.chart.set_rows([])
            self.noise.clear()
            self.sens.clear()
            self.suggestion_ready(False)
            self._redraw()
            return
        st = doc.get('settings') or {}
        s = self._sel(doc)
        bd = s['breakdown']
        worst = max(DV.REASONS, key=lambda k: bd.get(k, 0.0))
        per = '; '.join(f'{x["name"]} {x["valid_pct"]:.1f} %' for x in ss)
        sug = doc.get('suggestion')
        if sug:
            act = sug['text']
        elif bd.get(worst, 0) > 0.05:
            act = f'The largest loss for {s["name"]} is "{DV.REASON_TITLE[worst].lower()}" ({bd[worst]:.1f} %): ' \
                  f'{ADVICE[worst]}.'
        else:
            act = 'Nothing to improve within these conditions.'
        sev, word = (doc.get('rig') or {}).get('status', 'warning'), (doc.get('rig') or {}).get('status_word', '')
        tip = (f'Status: good from {st.get("good_pct", DV.DEFAULT_GOOD_PCT):g} %, partial from '
               f'{st.get("fair_pct", DV.DEFAULT_FAIR_PCT):g} % (defaults; settings good_pct and fair_pct). '
               f'Pose: {(doc.get("pose") or {}).get("note", "")}.')
        self.head.set(f'{rig:.1f}', '%', sev, word, f'Per sensor: {per}. {act}', tip)
        self.suggestion_ready(bool(sug))
        rows = [('valid', DV.REASON_TITLE['valid'], bd.get('valid', 0.0), 'all five conditions hold')]
        rows += [(k, DV.REASON_TITLE[k], bd.get(k, 0.0), DV.REASON_TEXT[k]) for k in DV.REASONS]
        self.chart.title = f'{s["name"]}: the first condition each voxel fails'
        self.chart.set_rows(rows)
        oc = s.get('occlusion') or {}
        self.occl_note.setText(
            f'Occluded (any imager, inside the field of view): {s["fails"]["occluded"]:.1f} % of the task volume; '
            f'hidden from one imager only (finger shadows, stereo parallax) {oc.get("one_imager_pct", 0):.1f} %, '
            f'from every imager {oc.get("all_imagers_pct", 0):.1f} %. {doc.get("voxels", {}).get("message", "")}')
        # fingertips
        tips = s.get('fingertips') or []
        inside = [t for t in tips if t['inside_min_z'] and t['in_fov']]
        if not tips:
            self.fingers.set('', 'No fingertip frames are tagged on the robots, so their depth is not checked.')
        elif inside:
            names = ', '.join(f'{t["link"]} at {t["z_m"] * 1000:.0f} mm' for t in inside)
            self.fingers.set('warning', f'Inside {s["name"]}\'s min-Z ({_mm(s["min_z_m"])}): {names}. The fingertips '
                                        'have no depth there: move the sensor back or use a resolution with a '
                                        'smaller min-Z.')
        else:
            names = ', '.join(f'{t["link"]} {t["z_m"] * 1000:.0f} mm' + ('' if t['in_fov'] else ' (out of view)')
                              for t in tips)
            self.fingers.set('good', f'Fingertips beyond {s["name"]}\'s min-Z ({_mm(s["min_z_m"])}): {names}.')
        ms = s.get('materials') or []
        self.mats.setText('\n'.join(f'Material warning · {m["object"]} ({m["pct"]:.2f} % of the task volume that '
                                    f'is otherwise depth-valid): {m["reason"]}.' for m in ms))
        self.mats.setVisible(bool(ms))
        self._fill_noise(ss)
        self._fill_sensors(ss)
        notes = list(s.get('notes') or [])
        notes += [i['label'] for i in s.get('imagers') or []]
        notes.append(f'σ_d: {s["sigma_d_note"]}.')
        if doc.get('skipped'):
            notes += [k['why'] for k in doc['skipped']]
        notes += (doc.get('occluders') or {}).get('notes') or []
        self.assume.setText('\n'.join(notes))
        self._redraw()

    def _fill_noise(self, ss):
        self.noise.clear()
        for s in ss:
            sz, mk = s.get('sigma_z'), s.get('maker_rms')
            sd = f'{s["sigma_d_px"]:g} px' + (' (default)' if s['sigma_d_default'] else '')
            if sz:
                zcol = f'{sz["median_m"] * 1000:.2f} mm [{sz["p5_m"] * 1000:.2f}–{sz["p95_m"] * 1000:.2f}]'
                rel = f'{sz["median_rel"] * 100:.2f} %'
            else:
                zcol, rel = '—', '—'
            model = maker = equiv = '—'
            if mk:
                if mk.get('model_sigma_m') is not None:
                    model = f'{mk["model_sigma_m"] * 1000:.1f} mm at {mk["at_m"]:g} m'
                if mk.get('sigma_m') is not None:
                    maker = (f'≤ {mk["sigma_m"] * 1000:.1f} mm {METRIC_WORDS.get(mk.get("metric"), "")}'.rstrip()
                             + f' ({mk["pct"]:g} % at {mk["at_m"]:g} m)')
                elif mk.get('text'):
                    maker = 'see note'
                if mk.get('sigma_d_equiv_px') is not None:
                    equiv = f'{mk["sigma_d_equiv_px"]:.3f} px'
                elif mk.get('sigma_m') is not None and mk.get('metric') != 'rms':
                    equiv = 'not an RMS'
            b = '—' if s.get('baseline_m') is None else f'{s["baseline_m"] * 1000:.1f} mm'
            vals = [s['name'], f'{s["resolution"][0]} × {s["resolution"][1]}' + (' (default)' if
                                                                                 s['resolution_default'] else ''),
                    f'{s["f_px"]:.1f} px', b, sd, zcol, rel, model, maker, equiv]
            it = QTreeWidgetItem(vals)
            for k in range(1, len(vals)):
                it.setFont(k, mono(11))
                it.setTextAlignment(k, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            it.setToolTip(1, s['resolution_note'])
            it.setToolTip(3, NOT_STATED if s.get('baseline_m') is None else 'As stated by the maker')
            it.setToolTip(4, s['sigma_d_note'])
            it.setToolTip(5, (sz or {}).get('interval', '') or s.get('sigma_z_why', ''))
            it.setToolTip(8, (mk or {}).get('text') or NOT_STATED)
            it.setToolTip(9, 'The disparity error for which the model reproduces the maker\'s figure'
                          if (mk or {}).get('metric') == 'rms' else
                          'Given only for an RMS figure: the maker\'s figure here is '
                          + METRIC_TIPS.get((mk or {}).get('metric'), METRIC_TIPS['unspecified']))
            self.noise.addTopLevelItem(it)
        _fit(self.noise)
        self.noise_note.setText(
            'σ_z = z² σ_d / (f B): f the depth imager\'s focal length in pixels at the chosen resolution, B the stated '
            'baseline, σ_d the matcher\'s subpixel disparity error (stated by the maker, else 0.08 px, a default). '
            'The maker\'s figure is measured its own way (an RMS against a fitted plane over most of the field of '
            'view, a median absolute error, or a metric it does not define; the tooltip quotes it), not the same '
            'quantity as σ_z: both are shown, and for an RMS figure the σ_d that would match it.')

    def _fill_sensors(self, ss):
        self.sens.clear()
        for s in ss:
            pj = s['projector']
            vals = [s['name'], s['title'], f'{s["valid_pct"]:.1f} %', _mm(s['min_z_m']),
                    '—' if s['max_range_m'] is None else f'{s["max_range_m"]:g} m',
                    'needed' if pj['needed'] else 'not needed',
                    f'{s["occlusion"]["one_imager_pct"]:.1f} %', f'{s["occlusion"]["all_imagers_pct"]:.1f} %']
            it = QTreeWidgetItem(vals)
            for k in (2, 3, 4, 6, 7):
                it.setFont(k, mono(11))
                it.setTextAlignment(k, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            it.setForeground(2, QColor({'good': '#0ca30c', 'warning': '#fab219'}.get(s['status'], '#d03b3b')))
            it.setToolTip(2, f'{s["status_word"]}: {s["valid_pct"]:.1f} % of the task volume is depth-valid')
            it.setToolTip(3, s['min_z_note'] if s['min_z_m'] is not None else NOT_STATED + ': ' + s['min_z_note'])
            it.setToolTip(4, s['max_range_note'])
            it.setToolTip(5, pj['note'])
            self.sens.addTopLevelItem(it)
        _fit(self.sens)

    # ---------------------------------------------------------------- Apply / Preview
    def apply(self):
        sug = (self.doc or {}).get('suggestion')
        if not sug:
            return
        self.set_resolution(sug['sensor_uid'], sug['resolution'])
        self._fill_controls()
        self.run()

    def preview(self):
        sug = (self.doc or {}).get('suggestion')
        self._preview = sug if sug and self._preview is None else None
        if self.ctx.window is not None:
            self.ctx.window.msg('Preview: the dashed outline is the min-Z zone at the suggested resolution' if
                                self._preview else 'Preview off', 6000)
            self.ctx.window.viewport.update()

    # ---------------------------------------------------------------- 3D: invalid voxels (this page)
    def _redraw(self):
        self._overlay_key = None
        if self.ctx.window is not None:
            self.ctx.window.viewport.update()

    def overlays(self, vp) -> list:
        doc = self.doc
        if not doc:
            return []
        s = self._sel(doc)
        if s is None:
            return []
        shown = tuple(k for k in DV.REASONS if self.show[k].isChecked())
        key = (id(doc), s['uid'], shown)
        if key == self._overlay_key:
            return self._overlay_items
        ov = s.get('overlay') or {}
        P = np.asarray(ov.get('points_mm') or np.zeros((0, 3)), float).reshape(-1, 3) / 1000.0
        R = np.asarray(ov.get('reason') or [], int)
        h = 0.35 * float(ov.get('voxel_mm', 5.0)) / 1000.0
        items = []
        for j, k in enumerate(DV.REASONS, start=1):
            if k not in shown:
                continue
            Q = P[R == j]
            if len(Q) > OVERLAY_DRAW_MAX:
                Q = Q[np.linspace(0, len(Q) - 1, OVERLAY_DRAW_MAX).astype(int)]
            if not len(Q):
                continue
            segs = []
            for ax in range(3):
                d = np.zeros(3)
                d[ax] = h
                segs.append(np.stack([Q - d, Q + d], 1))
            items.append(LineDraw(np.concatenate(segs), QColor(REASON_COLOR[k]), 1.2, segments=True))
        self._overlay_key, self._overlay_items = key, items
        return items

    # ---------------------------------------------------------------- Camera view: the other sensors' min-Z zones
    def camview_overlays(self, cam_model, pose) -> list:
        """Each other depth sensor's min-Z blind zone (where it gives no depth), outlined in the Camera view: a wrist
        sensor's zone seen from the overhead camera shows how close to the fingers depth is lost."""
        from ...core import camrender as CR
        from ...core import scenegeom as SG
        p = self.ctx.store.project
        cv = getattr(self.ctx.window, 'camview', None) if self.ctx.window is not None else None
        shown = cv.view.sensor() if cv is not None and hasattr(cv, 'view') else None
        out = []
        for s in self._sensors():
            if shown is not None and s.uid == shown.uid:
                continue                                # its own zone is the frame border: nothing to see
            geo = self._zone_geo(p, s)
            if geo is None:
                continue
            try:
                M = SG.sensor_world_pose(p, s, self.ctx.store.q).matrix()
            except Exception:                           # noqa: BLE001 — a sensor that cannot be posed is not drawn
                continue
            segs = geo[1] @ M[:3, :3].T + M[:3, 3]
            polys = []
            for seg in segs:
                polys += CR.visible_polylines(None, cam_model, pose, seg, step=0.005)
            if polys:
                out.append((polys, THEME.cam(s.slot), f'min-Z zone · {s.name}'))
        return out

    # ---------------------------------------------------------------- 3D: min-Z zones (every page)
    def _zone_geo(self, project, s, res=None):
        cfg = DV.sensor_settings(project, s.uid)
        if res is not None:
            cfg['resolution'] = list(res)
        from ...core import library
        # the zone depends on the device's figures (a custom device can change under its id), its calibration, the
        # study's resolution setting and the sensor's chosen stream (the resolution when the setting is absent)
        key = (s.model_id, library.generation(), repr(sorted((s.calibration or {}).items())),
               repr(cfg.get('resolution')), tuple(getattr(s, 'stream', ()) or ()))
        if key not in self._zone_cache:
            try:
                ds = DV.setup(project, s, cfg)
                self._zone_cache[key] = DV.blind_zone_local(ds)
            except DV.DepthSetupError:
                self._zone_cache[key] = None
        return self._zone_cache[key]

    def zones(self, vp) -> list:
        """Every depth sensor's min-Z blind zone at the joint state on screen (an overlay source)."""
        from ...core import scenegeom as SG
        if not self.ctx.overlay('depthzones'):
            return []
        p = self.ctx.store.project
        out = []
        prev = self._preview
        for s in self._sensors():
            geo = self._zone_geo(p, s)
            alt = self._zone_geo(p, s, prev['resolution']) if prev and prev.get('sensor_uid') == s.uid else None
            if geo is None and alt is None:
                continue
            try:
                body = SG.sensor_world_pose(p, s, self.ctx.store.q)
            except Exception:                           # noqa: BLE001 — a sensor that cannot be posed is not drawn
                continue
            M = body.matrix()
            R, t = M[:3, :3], M[:3, 3]
            if geo is not None:
                tris, segs = geo
                c = np.array(THEME.cam(s.slot).getRgbF())
                c[3] = ZONE_ALPHA
                out.append(TriDraw(tris @ R.T + t, np.tile(c, (len(tris), 1)), depth=True))
                out.append(LineDraw(segs @ R.T + t, THEME.cam(s.slot, 0.8), 1.2, segments=True))
            if alt is not None:
                out.append(LineDraw(alt[1] @ R.T + t, THEME.color('ink', 0.9), 1.6, dash=True, segments=True,
                                    dash_len=0.01))
        return out


register('depth', 'Depth validity', DepthPage, menu='Analysis', icon='depth', action_id='depth_validity',
         workspace='placement', order=20)
