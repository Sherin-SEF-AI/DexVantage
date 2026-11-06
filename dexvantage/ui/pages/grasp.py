"""The Grasp self-occlusion page (spec §7.3, the hero study): of the grasps in a grasp set, how many have their contact
in view of the rig's cameras, and which camera to move to raise that.

What it shows: the Contact Visibility Score (the share of the reachable grasps where the requirement holds, default
every fingertip at least 50 % visible in at least one camera, labelled as a default) with its bootstrap interval and
a status word from the user's thresholds (defaults 90 %, 70 % and 40 %); the reachability (infeasible grasps by
reason, and the obstacle pairs); a table per camera (each fingertip's visible share, the both-fingertips rate, the
object, the pre-contact visibility, and why contact points are not seen); "what to move" (one camera at a time moved
or turned by a small step, ranked by the change in the score, with its paired interval); the worst-case gallery
(Camera view renders of the lowest-scoring grasps with each contact point marked); and the grasps, lowest score
first. The 3D view shows the grasp centres coloured by score (the heatmap over the task volume), and the Camera view
the same centres in the image. Apply moves the camera of the best move (one undo step); Preview draws where it would
go. Clicking a grasp or a gallery image puts the robot in that grasp's pose.
"""
from __future__ import annotations

import json
import traceback

import numpy as np
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QHeaderView, QLabel, QMessageBox, QTreeWidget,
                             QTreeWidgetItem, QVBoxLayout, QWidget)

from ...core.studies import grasp_visibility as GV
from .. import colormaps as CM
from ..theme import STATUS, THEME, mono
from ..widgets import ScrubSpinBox, label
from . import register
from .base import StudyPage, row

MAX_ROWS = 300                 # grasps listed (lowest score first); Export holds every grasp
MAX_CAM_POINTS = 600           # grasp centres drawn in the Camera view
GLYPH_M = 0.006                # half-size of a heatmap glyph (m)
THUMB_W = 220
STATE_COLORS = {GV.ST_OUT: '#8a8a85', GV.ST_AWAY: STATUS['serious'], GV.ST_HIDDEN: STATUS['critical'],
                GV.ST_SEEN: STATUS['good']}
SETTINGS = (('min_visible', 'Fingertip visible from', '%', 100.0, 'A fingertip counts as seen when at least this '
             'share of its contact patch is visible (default 50 %).'),
            ('min_cameras', 'Cameras needed', '', 1.0, 'How many cameras must see it (default 1).'),
            ('good_pct', 'Good from', '%', 1.0, 'The score that reads "good" (default 90 %).'),
            ('warn_pct', 'Warning from', '%', 1.0, 'Below "good" and from this score it reads "warning" (default '
                                                    '70 %).'),
            ('serious_pct', 'Serious from', '%', 1.0, 'Below "warning" and from this score it reads "serious"; below '
                                                      'it "critical" (default 40 %).'))
REQ_LABELS = [('all_fingertips', 'Every fingertip'), ('any_fingertip', 'Any fingertip'),
              ('object', 'The object region'), ('fingertips_and_object', 'Fingertips and object')]
RULE_LABELS = [('nearest_home', 'Nearest to home'), ('most_manipulable', 'Most manipulable')]


def _pct(x, nd=0) -> str:
    return '—' if x is None else f'{100 * x:.{nd}f} %'


def _f(x, fmt='{:.2f}') -> str:
    return '—' if x is None else fmt.format(x)


def score_colors(score, min_visible: float) -> np.ndarray:
    """(N, 4) RGBA 0..1: a diverging ramp centred on min_visible (orange below, blue above), grey for NaN."""
    lut = CM.diverging(n=257).astype(float) / 255.0
    s = np.asarray(score, float)
    mv = min(max(float(min_visible), 1e-6), 1 - 1e-6)
    t = np.where(s < mv, 0.5 * s / mv, 0.5 + 0.5 * (s - mv) / (1 - mv))
    k = np.clip(np.nan_to_num(t, nan=0.5) * 256, 0, 256).astype(int)
    out = np.c_[lut[k], np.ones(len(s))]
    out[~np.isfinite(s)] = (0.55, 0.55, 0.53, 0.6)
    return out


def octahedra(P, h: float) -> np.ndarray:
    """(8N, 3, 3) triangles of small octahedra of half-size h at points P (N, 3)."""
    P = np.asarray(P, float).reshape(-1, 3)
    e = np.eye(3) * h
    V = [e[0], -e[0], e[1], -e[1], e[2], -e[2]]
    faces = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    T = np.array([[V[a], V[b], V[c]] for a, b, c in faces])           # (8, 3, 3)
    return (P[:, None, None, :] + T[None]).reshape(-1, 3, 3).astype(np.float32)


class GraspPage(StudyPage):
    """Grasp self-occlusion: for every grasp of the chosen set the arm is solved by inverse kinematics (reachable and
    free of collisions), and at that pose each camera is asked which points of each fingertip's contact patch (and
    of the object's grasp region) it sees: inside its image, and nothing of the posed robot, the objects or the tables
    in front (a cube map of distances around the camera). The Contact Visibility Score is the share of the reachable
    grasps where your requirement holds, with a bootstrap interval."""
    STUDY = GV.SID
    TITLE = 'Grasp self-occlusion'
    HEADLINE = 'Contact Visibility Score'

    # ---------------------------------------------------------------- body
    def build_body(self, layout):
        self.dec = None
        self._overlay = []
        self._preview = []
        self._spin = {}
        self._inputs = None
        # what to study
        self.set_box = QComboBox()
        self.set_box.setToolTip('The grasp set to study (make one on the Tasks page)')
        self.set_box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.req_box = QComboBox()
        for k, t in REQ_LABELS:
            self.req_box.addItem(t, k)
        self.req_box.setToolTip('What must be seen for a grasp to count (default: every fingertip)')
        self.rule_box = QComboBox()
        for k, t in RULE_LABELS:
            self.rule_box.addItem(t, k)
        self.rule_box.setToolTip('Which IK solution to keep when several reach the grasp (default: nearest to home)')
        self.cam_box = QComboBox()
        self.cam_box.setToolTip('The cameras the score is for (default: every camera and depth sensor in use)')
        self.cam_box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.cam_box.activated.connect(lambda _i: self._set_setting(
            'sensors', [self.cam_box.currentData()] if self.cam_box.currentData() else None, 'cameras'))
        self.set_box.activated.connect(lambda _i: self._set_setting('grasp_set', self.set_box.currentData() or None,
                                                                    'grasp set'))
        self.req_box.activated.connect(lambda _i: self._set_setting('requirement', self.req_box.currentData(),
                                                                    'requirement'))
        self.rule_box.activated.connect(lambda _i: self._set_setting('rule', self.rule_box.currentData(), 'IK rule'))
        layout.addWidget(row(label('Grasp set'), self.set_box, label('Cameras'), self.cam_box))
        layout.addWidget(row(label('Requirement'), self.req_box, label('IK'), self.rule_box))
        g = QWidget()
        grid = QGridLayout(g)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        for i, (key, text, unit, scale, tip) in enumerate(SETTINGS):
            sb = ScrubSpinBox('', 'plain', 0 if key == 'min_cameras' else 1, unit, step=1.0 if key == 'min_cameras'
                              else 5.0, width=100)
            sb.setRange(1.0 if key == 'min_cameras' else 0.0, 16.0 if key == 'min_cameras' else 100.0)
            sb.setToolTip(tip)
            dl = label('', muted=True, small=True)
            grid.addWidget(label(text), i // 3, (i % 3) * 3)
            grid.addWidget(sb, i // 3, (i % 3) * 3 + 1)
            grid.addWidget(dl, i // 3, (i % 3) * 3 + 2)
            self._spin[key] = (sb, dl, scale)
            sb.committed.connect(lambda v, k=key: self._set_number(k, v))
        layout.addWidget(g)
        self.reach_label = label('', muted=True, wrap=True)
        layout.addWidget(self.reach_label)
        self.stale_note = label('', muted=True, wrap=True)
        layout.addWidget(self.stale_note)

        layout.addWidget(label('Cameras', bold=True))
        self.cams = QTreeWidget()
        self.cams.setRootIsDecorated(False)
        self.cams.setHeaderLabels(['Camera', 'Fingertips visible', 'Both ≥ threshold', 'Object', 'Pre-contact',
                                   'Seen', 'Outside image', 'Behind own finger', 'Hidden by scene'])
        _columns(self.cams)
        layout.addWidget(self.cams)

        layout.addWidget(label('What to move (one camera at a time)', bold=True))
        self.moves = QTreeWidget()
        self.moves.setRootIsDecorated(False)
        self.moves.setHeaderLabels(['#', 'Change', 'Score after', 'Change in score', 'Interval', 'Visible share'])
        _columns(self.moves, stretch=1)
        layout.addWidget(self.moves)

        layout.addWidget(label('Worst grasps (Camera view renders)', bold=True))
        self.gallery = QWidget()
        self.gallery_grid = QGridLayout(self.gallery)
        self.gallery_grid.setContentsMargins(0, 0, 0, 0)
        self.gallery_grid.setSpacing(6)
        layout.addWidget(self.gallery)
        self.gallery_legend = label('', muted=True, small=True, wrap=True)
        layout.addWidget(self.gallery_legend)

        self.grasps_title = label('Grasps, lowest score first', bold=True)
        layout.addWidget(self.grasps_title)
        self.grasps = QTreeWidget()
        self.grasps.setRootIsDecorated(False)
        self.grasps.setUniformRowHeights(True)
        self.grasps.setMinimumHeight(200)
        self.grasps.itemClicked.connect(lambda it, _c: self.show_grasp(it.data(0, Qt.ItemDataRole.UserRole)))
        layout.addWidget(self.grasps)

        self.show3d = QCheckBox('Show the grasp centres in the 3D view, coloured by score')
        self.show3d.setChecked(True)
        self.show3d.toggled.connect(lambda _on: self._redraw())
        layout.addWidget(self.show3d)
        self.legend = label('', muted=True, small=True, wrap=True)
        layout.addWidget(self.legend)
        self.notes_label = label('', muted=True, wrap=True)
        self.notes_label.setVisible(False)
        layout.addWidget(self.notes_label)
        self.perf_label = label('', muted=True, small=True, wrap=True)
        layout.addWidget(self.perf_label)
        st = self.ctx.store
        st.changed.connect(lambda _s: (self._sync(), self._check_stale()))
        st.project_replaced.connect(self._replaced)
        self.ctx.analysis.partial.connect(self._partial)
        self._sync()

    # ---------------------------------------------------------------- settings
    def settings(self) -> dict:
        try:
            return GV.settings_of(self.ctx.store.project)
        except Exception:                               # noqa: BLE001 — a bad stored value: show the defaults
            return dict(GV.DEFAULTS, defaults=list(GV.DEFAULTS), grasp_set='')

    def _sync(self):
        from ...core import grasps as GR
        st = self.settings()
        p = self.ctx.store.project
        self.set_box.blockSignals(True)
        self.set_box.clear()
        for gs in GR.sets(p):
            r = p.robot(gs.get('robot_uid', ''))
            self.set_box.addItem(f'{gs.get("name")} · {len(gs.get("poses") or [])} grasps' +
                                 (f' · {r.name}' if r is not None else ''), gs.get('uid'))
        if not self.set_box.count():
            self.set_box.addItem('No grasp set yet (Tasks page)', '')
        k = self.set_box.findData(st.get('grasp_set') or '')
        self.set_box.setCurrentIndex(k if k >= 0 else 0)
        self.set_box.blockSignals(False)
        self.cam_box.blockSignals(True)
        self.cam_box.clear()
        self.cam_box.addItem('All cameras in use', '')
        for sn in p.variant.sensors:
            if sn.kind in ('camera', 'depth'):
                self.cam_box.addItem(sn.name + ('' if sn.enabled else ' (not in use)'), sn.uid)
        chosen = st.get('sensors') or []
        k = self.cam_box.findData(chosen[0]) if len(chosen) == 1 else 0
        self.cam_box.setCurrentIndex(k if k >= 0 else 0)
        self.cam_box.blockSignals(False)
        for box, key in ((self.req_box, 'requirement'), (self.rule_box, 'rule')):
            box.blockSignals(True)
            k = box.findData(st.get(key))
            box.setCurrentIndex(k if k >= 0 else 0)
            box.blockSignals(False)
        for key, (sb, dl, scale) in self._spin.items():
            v = float(st.get(key, GV.DEFAULTS[key]))
            sb.set_value(v * (100.0 if key == 'min_visible' else 1.0))
            d = GV.DEFAULTS[key]
            dl.setText('default' if abs(v - float(d)) < 1e-12 else
                       f'yours (default {d * (100 if key == "min_visible" else 1):g})')

    def _set_setting(self, key, value, what):
        cur = (self.ctx.store.project.analysis.studies.get(GV.SID) or {}).get(key)
        if cur == value:
            return

        def fn(p):
            d = dict(p.analysis.studies.get(GV.SID) or {})
            if value is None:
                d.pop(key, None)
            else:
                d[key] = value
            p.analysis.studies[GV.SID] = d
        self.ctx.store.edit('analysis', fn, f'Grasp study · {what}', merge_key=(GV.SID, key))

    def _set_number(self, key, v):
        val = float(v) / (100.0 if key == 'min_visible' else 1.0)
        if key == 'min_cameras':
            val = int(round(val))
        if abs(float(self.settings().get(key, 0)) - val) < 1e-12:
            return
        text = next(t for k, t, _u, _s, _tip in SETTINGS if k == key)
        self._set_setting(key, val, f'{text.lower()} {v:g}')
        if self.doc is not None and key in ('good_pct', 'warn_pct', 'serious_pct'):
            self._headline(self.doc)

    # ---------------------------------------------------------------- run
    def live_q(self) -> dict:
        return {uid: [float(x) for x in np.asarray(q, float).reshape(-1)] for uid, q in self.ctx.store.q.items()}

    def run(self):
        """Run on the joint state on screen for the robots the grasps do not move (passed in the run's settings)."""
        self.stale_note.setText('')
        self.ctx.analysis.run(self.STUDY, {'q': self.live_q()})

    def _partial(self, sid, doc):
        if sid != self.STUDY or not isinstance(doc, dict):
            return
        v = doc.get('cvs_so_far')
        self.note.setText(f'Running in the background · {doc.get("evaluated", 0):,} of {doc.get("feasible", 0):,} '
                          'reachable grasps' + (f' · score so far {_pct(v)}' if v is not None else '') +
                          (f' · {doc["grasps_per_s"]:.1f} grasps/s' if doc.get('grasps_per_s') else ''))

    def _failed(self, sid, message):
        if sid == self.STUDY:
            from ..store import failure_text
            first, _explained = failure_text(message, 'The grasp study')
            self.head.set('—', '', 'critical', 'could not run', first)

    def _replaced(self):
        self.doc, self.dec, self._overlay, self._preview, self._inputs = None, None, [], [], None
        for t in (self.cams, self.moves, self.grasps):
            t.clear()
        self._clear_gallery()
        self.notes_label.setVisible(False)
        self.stale_note.setText('')
        self.reach_label.setText('')
        self.perf_label.setText('')
        self.suggestion_ready(False)
        self._sync()
        self.empty()

    def inputs_key(self) -> str:
        p = self.ctx.store.project
        st = {k: v for k, v in (p.analysis.studies.get(GV.SID) or {}).items()
              if k not in ('good_pct', 'warn_pct', 'serious_pct', 'q')}
        return json.dumps([p.workspace.to_dict(), p.variant.to_dict(), [r.to_dict() for r in p.robots],
                           p.tasks.grasp_sets, st], sort_keys=True, default=str)

    def _check_stale(self):
        if self.doc is None or self._inputs is None:
            return
        self.stale_note.setText('The project changed since this run: Run again to update the score.'
                                if self.inputs_key() != self._inputs else '')

    # ---------------------------------------------------------------- result
    def _headline(self, doc):
        s = doc.get('summary') or {}
        if s.get('cvs') is None:
            self.head.set('—', '', 'critical', 'nothing to score', s.get('advice') or 'No grasp could be scored.')
            return
        st = self.settings()
        sev, word = GV.status_for(100 * s['cvs'], st)
        ci = ''
        point = s.get('ci_low') is not None and s['ci_low'] == s['ci_high']
        if point and s.get('wilson_low') is not None:               # every resample alike: Wilson's says more
            ci = f' [{100 * s["wilson_low"]:.1f}–{100 * s["wilson_high"]:.0f}]'
        elif s.get('ci_low') is not None:
            ci = f' [{100 * s["ci_low"]:.0f}–{100 * s["ci_high"]:.0f}]'
        lvl = f'{100 * s.get("ci_level", 0.95):.0f} %'
        tip = (f'{s["meets"]:,} of {s["evaluated"]:,} reachable grasps have {s.get("requirement", "")}. Interval: '
               f'{lvl} {s.get("ci_method", "")}; Wilson\'s score interval {_pct(s.get("wilson_low"), 1)} to '
               f'{_pct(s.get("wilson_high"), 1)}. Of all {s["grasps"]:,} grasps (unreachable ones count as failing): '
               f'{_pct(s.get("cvs_of_all"), 1)}. Status: good from {st["good_pct"]:g} %, warning from '
               f'{st["warn_pct"]:g} %, serious from {st["serious_pct"]:g} % (DexVantage defaults 90, 70 and 40 %, '
               'not standards).')
        explain = s.get('advice', '')
        if point:
            explain += (f' Every resample gives the same score, so the bootstrap interval is a point; the figure shows '
                        f'Wilson\'s {lvl} score interval instead, {_pct(s.get("wilson_low"), 1)} to '
                        f'{_pct(s.get("wilson_high"), 1)}.')
        self.head.set(f'{100 * s["cvs"]:.1f}{ci}', '% of reachable grasps', sev, word, explain, tip)

    def show_result(self, doc):
        if doc.get('partial') and not doc.get('grasps'):
            return
        t_ok = True
        try:
            self._headline(doc)
            self._fill_reach(doc)
            self._fill_cams(doc)
            self._fill_moves(doc)
            self._fill_gallery(doc)
            self.dec = GV.decode(doc) if doc.get('grasps') else None
            self._fill_grasps(doc)
            self._inputs = self.inputs_key()
        except (KeyError, ValueError, TypeError):
            traceback.print_exc()
            t_ok = False
        notes = [str(n) for n in doc.get('notes') or []]
        self.notes_label.setText('\n'.join(notes))
        self.notes_label.setVisible(bool(notes))
        self.suggestion_ready(t_ok and doc.get('best_move') is not None)
        self._preview = []
        self._build_overlay()
        self._check_stale()
        self._refresh_camview()

    def _fill_reach(self, doc):
        s = doc['summary']
        inf = s.get('infeasible') or {}
        parts = [f'{s["feasible"]:,} of {s["grasps"]:,} grasps reachable without a collision']
        if inf:
            parts.append('unreachable: ' + ', '.join(f'{v:,} {k}' for k, v in sorted(inf.items(), key=lambda t: -t[1])))
        pairs = (doc.get('ik') or {}).get('obstacle_pairs') or {}
        if pairs:
            parts.append('most obstacle contacts: ' + ', '.join(f'{k} ({v})' for k, v in list(pairs.items())[:3]))
        rs = doc.get('grasp_set') or {}
        self.reach_label.setText(f'{rs.get("name", "")}: ' + '; '.join(parts) + '.')
        th, oc = doc.get('throughput') or {}, doc.get('occlusion') or {}
        tm = doc.get('timings') or {}
        self.perf_label.setText(
            f'{s["grasps"]:,} grasps in {tm.get("total_seconds", 0):.1f} s ({_f(th.get("grasps_per_s"), "{:.1f}")} '
            f'grasps/s; visibility {_f(th.get("occlusion_grasps_per_s"), "{:.1f}")} grasps/s). '
            f'{oc.get("maps_built", 0):,} cube maps ({oc.get("cube_n")} texels, '
            f'{oc.get("renderer") or oc.get("backend")}), cache hit rate {_pct(oc.get("cache_hit_rate"))} '
            f'(q-bucket {oc.get("q_bucket_deg", 0):g}°, a default).')

    def _fill_cams(self, doc):
        self.cams.clear()
        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        for c in doc.get('cameras', []):
            tips = ', '.join(_pct(v) for v in (c.get('fingertips') or {}).values())
            pts = c.get('points') or {}
            it = QTreeWidgetItem([c['name'], tips, _pct(c.get('both_fingertips_rate')), _pct(c.get('object_visible')),
                                  _pct(c.get('precontact_mean')), _pct(pts.get('visible')),
                                  _pct(pts.get('outside_image')), _pct(pts.get('behind_own_finger')),
                                  _pct(pts.get('hidden_by_scene'))])
            it.setToolTip(0, f'{c.get("imager_label", "")}. Used: {c.get("imager_why", "")}.')
            it.setToolTip(1, 'Mean visible share of each fingertip\'s contact patch over the reachable grasps (' +
                          ', '.join((c.get('fingertips') or {}).keys()) + ')')
            it.setToolTip(2, 'Share of the reachable grasps where this camera sees every fingertip at or above the '
                             'threshold')
            it.setToolTip(4, 'Mean visible share of the weaker fingertip over the last part of the approach '
                             f'({", ".join(_f(d * 100, "{:.1f} cm") for d in doc.get("precontact_m") or [])})')
            it.setToolTip(7, 'Contact points on the far side of their own finger: the face is turned away')
            it.setToolTip(8, 'Contact points hidden by the robot, the objects or the tables')
            for k in range(1, 9):
                it.setFont(k, mono(11))
                it.setTextAlignment(k, right)
            self.cams.addTopLevelItem(it)
        _fit(self.cams)

    def _fill_moves(self, doc):
        self.moves.clear()
        best = doc.get('best_move')
        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        for m in (doc.get('sensitivity') or [])[:12]:
            d = m.get('delta_cvs')
            iv = '—' if m.get('delta_low') is None else f'{100 * m["delta_low"]:+.0f} to {100 * m["delta_high"]:+.0f}'
            it = QTreeWidgetItem([str(m['rank']), m['text'], _pct(m.get('cvs')),
                                  '—' if d is None else f'{100 * d:+.1f} pts', iv,
                                  _f(m.get('delta_score'), '{:+.3f}')])
            it.setToolTip(1, f'Measured on {m.get("grasps", 0)} reachable grasps (a seeded subset), the other cameras '
                             'unchanged.')
            it.setToolTip(4, 'Paired bootstrap interval of the change in score')
            it.setToolTip(5, 'Mean change of each grasp\'s score (the weaker fingertip\'s visible share in the best '
                             'camera): it shows a direction even where the score does not change')
            if best is not None and m is not None and m.get('text') == best.get('text'):
                for k in range(6):
                    f = it.font(k)
                    f.setBold(True)
                    it.setFont(k, f)
            for k in (0, 2, 3, 4, 5):
                it.setFont(k, mono(11))
                it.setTextAlignment(k, right)
            self.moves.addTopLevelItem(it)
        _fit(self.moves)

    def _clear_gallery(self):
        while self.gallery_grid.count():
            w = self.gallery_grid.takeAt(0).widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        for c in range(self.gallery_grid.columnCount()):
            self.gallery_grid.setColumnStretch(c, 0)
        self._gallery_cells, self._gallery_per = [], 0

    def _gallery_fit(self) -> int:
        """How many grasps fit side by side in the page's width (each grasp its cameras' thumbnails, then a gap)."""
        cells = getattr(self, '_gallery_cells', [])
        if not cells:
            return 1
        ncam = max(j for _g, j, _c in cells) + 1
        avail = max(self.body.width(), self.width()) - 40
        return max(1, (avail + 18) // (ncam * (THUMB_W + 6) + 18))

    def _place_gallery(self):
        """Lay the gallery's cells out: grasps flow left to right, each grasp's cameras kept together."""
        cells = getattr(self, '_gallery_cells', [])
        per = self._gallery_fit()
        if per == self._gallery_per and self.gallery_grid.count() == len(cells):
            return
        self._gallery_per = per
        for _g, _j, cell in cells:
            self.gallery_grid.removeWidget(cell)
        for c in range(self.gallery_grid.columnCount()):
            self.gallery_grid.setColumnStretch(c, 0)
        if not cells:
            return
        ncam = max(j for _g, j, _c in cells) + 1
        order = {gi: n for n, gi in enumerate(dict.fromkeys(g for g, _j, _c in cells))}
        stride = ncam + 1                                   # an empty spacer column between grasps
        for gi, j, cell in cells:
            n = order[gi]
            col = (n % per) * stride + j
            self.gallery_grid.addWidget(cell, n // per, col, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            cell.show()
        for m in range(per - 1):
            self.gallery_grid.setColumnMinimumWidth(m * stride + ncam, 12)
        last = per * stride - 1
        self.gallery_grid.setColumnStretch(last, 1)          # thumbnails packed to the left

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if getattr(self, '_gallery_cells', None) and self._gallery_fit() != self._gallery_per:
            self._place_gallery()

    def _fill_gallery(self, doc):
        self._clear_gallery()
        k = 0
        self._gallery_cells, self._gallery_per = [], 0
        for gi, g in enumerate(doc.get('gallery') or []):             # each grasp's cameras side by side
            for j, v in enumerate(g.get('views') or []):
                try:
                    pm = gallery_pixmap(v, THUMB_W)
                except (KeyError, ValueError):
                    traceback.print_exc()
                    continue
                lab = QLabel()
                lab.setPixmap(pm)
                lab.setToolTip(f'Grasp {g["id"]} ({g["kind"]}), score {_f(g.get("score"))}: {v["camera"]} sees '
                               + ', '.join(_pct(x) for x in v.get('fingertips') or []) + ' of the fingertips. Click to '
                               'put the robot in this pose.')
                lab.setCursor(Qt.CursorShape.PointingHandCursor)
                lab.mousePressEvent = (lambda _ev, i=g['index']: self.show_grasp(i))
                cap = label(f'{g["id"]} · {_f(g.get("score"))} · {v["camera"]}', muted=True, small=True)
                cell = QWidget()
                lay = QVBoxLayout(cell)
                lay.setContentsMargins(0, 0, 0, 0)
                lay.setSpacing(2)
                lay.addWidget(lab)
                lay.addWidget(cap)
                self._gallery_cells.append((gi, j, cell))
                k += 1
        self._place_gallery()
        self.gallery_legend.setText('Contact points: green visible, red hidden by the scene, orange behind their own '
                                    'finger, grey outside the image.' if k else 'No gallery for this run.')

    def _fill_grasps(self, doc):
        self.grasps.clear()
        t = doc.get('grasps') or {}
        if not t or self.dec is None:
            return
        cams = [c['name'] for c in doc.get('cameras', [])]
        self.grasps.setHeaderLabels(['Grasp', 'Kind', 'Reachable', 'Score', 'Meets'] +
                                    [f'{n} (weaker tip)' for n in cams] + ['Object (best)'])
        _columns(self.grasps)
        sc = self.dec['score']
        order = np.argsort(np.where(np.isfinite(sc), sc, np.inf), kind='stable')[:MAX_ROWS]
        frac = self.dec['fingertips']
        obj = self.dec['object']
        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        items = []
        for i in order:
            reach = 'yes' if t['feasible'][i] else (t['reason'][i] or 'no')
            per = [('—' if not t['feasible'][i] else _pct(float(np.nanmin(frac[i, j])))) for j in range(len(cams))]
            ob = obj[i]
            it = QTreeWidgetItem([t['id'][i], t['kind'][i], reach, _f(t['score'][i]),
                                  ('yes' if t['meets'][i] else 'no') if t['feasible'][i] else '—'] + per +
                                 [_pct(float(np.nanmax(ob))) if np.isfinite(ob).any() else '—'])
            it.setData(0, Qt.ItemDataRole.UserRole, int(i))
            for k in range(3, len(cams) + 6):
                it.setFont(k, mono(11))
                it.setTextAlignment(k, right)
            items.append(it)
        self.grasps.addTopLevelItems(items)
        n = len(t['id'])
        self.grasps_title.setText(f'Grasps, lowest score first ({min(n, MAX_ROWS):,} of {n:,}; Export holds every '
                                  'grasp)' if n > MAX_ROWS else 'Grasps, lowest score first')

    # ---------------------------------------------------------------- show a grasp, apply, preview
    def show_grasp(self, i):
        """Put the robot in grasp i's IK pose (view state, not an edit)."""
        if self.dec is None or i is None or self.doc is None:
            return
        q = self.dec['q'][int(i)]
        if not np.isfinite(q).all():
            if self.ctx.window is not None:
                self.ctx.window.msg('This grasp is not reachable, so there is no pose to show.', 4000)
            return
        try:
            self.ctx.store.set_q(self.doc['robot']['uid'], q)
        except Exception:                               # noqa: BLE001 — the robot may have gone
            traceback.print_exc()

    def apply(self):
        """Move the camera of the best move (one undo step)."""
        from ...core.frames import Pose
        best = (self.doc or {}).get('best_move')
        if not best:
            return None
        uid = best['sensor_uid']
        if self.ctx.store.project.variant.sensor(uid) is None:
            if self.ctx.window is not None:
                self.ctx.window.msg('That camera is not in the rig any more: Run again.', 5000)
            return None
        pose = Pose.from_dict(best['mount_pose'])

        def fn(p):
            s = p.variant.sensor(uid)
            s.mount.pose = Pose(pose.position.copy(), pose.quaternion.copy())
        self.ctx.store.edit('variant', fn, best['text'][0].upper() + best['text'][1:])
        self._preview = []
        self.suggestion_ready(False)
        self.stale_note.setText('The camera moved: Run again for the score with it there.')
        self._redraw()
        return uid

    def preview(self):
        """Draw where the best move would put its camera: a line from the camera to there and its new axis."""
        from ...core import scenegeom as SG
        from ...core.frames import Pose
        from ..viewport3d import LineDraw
        best = (self.doc or {}).get('best_move')
        p = self.ctx.store.project
        s = p.variant.sensor(best['sensor_uid']) if best else None
        if s is None:
            return
        qd = {u: np.asarray(q, float) for u, q in self.ctx.store.q.items()}
        try:
            now = SG.sensor_world_pose(p, s, qd)
            tmp = s.mount.pose
            s.mount.pose = Pose.from_dict(best['mount_pose'])
            try:
                then = SG.sensor_world_pose(p, s, qd)
            finally:
                s.mount.pose = tmp
        except Exception:                               # noqa: BLE001
            traceback.print_exc()
            return
        a, b = now.position, then.position
        ax = np.asarray(then.matrix()[:3, 2], float)
        col = QColor(STATUS['good'])
        self._preview = [LineDraw(np.array([[a, b]]), col, 2.0, dash=True, depth=False, segments=True),
                         LineDraw(np.array([[b, b + 0.08 * ax]]), col, 2.0, depth=False, segments=True)]
        self._redraw()
        if self.ctx.window is not None:
            self.ctx.window.msg(f'Preview: {best["text"]}. Apply makes the change.', 6000)

    # ---------------------------------------------------------------- overlays
    def _build_overlay(self):
        from ..viewport3d import TriDraw
        self._overlay = []
        if self.dec is None or self.doc is None:
            self.legend.setText('')
            return
        try:
            mv = float(self.doc['settings'].get('min_visible', 0.5))
            cols = score_colors(self.dec['score'], mv)
            tri = octahedra(self.dec['tcp'], GLYPH_M)
            self._overlay = [TriDraw(tri, np.repeat(cols, 8, axis=0).astype(np.float32), depth=True)]
            self.legend.setText(f'Grasp centres: blue at or above {100 * mv:.0f} % (the requirement holds), orange '
                                'below, darker further from it; grey unreachable.')
        except Exception:                               # noqa: BLE001 — the overlay must not take the page down
            traceback.print_exc()
            self._overlay = []
        self._redraw()

    def _redraw(self):
        win = self.ctx.window
        if win is not None and hasattr(win, 'viewport'):
            win.viewport.update()

    def _refresh_camview(self):
        cv = getattr(self.ctx.window, 'camview', None) if self.ctx.window is not None else None
        if cv is not None and hasattr(cv, 'view'):
            cv.view.refresh_overlays()

    def overlays(self, vp) -> list:
        return (list(self._overlay) if self.show3d.isChecked() else []) + list(self._preview)

    def camview_overlays(self, cam_model, pose) -> list:
        """The grasp centres in the Camera view: small crosses, by whether the requirement holds."""
        from ...core import camrender as CR
        if self.dec is None or not self.show3d.isChecked():
            return []
        P = self.dec['tcp']
        idx = np.arange(len(P))
        if len(idx) > MAX_CAM_POINTS:
            idx = np.unique(np.linspace(0, len(P) - 1, MAX_CAM_POINTS).round().astype(int))
        uv, ok, _d = CR.project_world(cam_model, pose, P[idx])
        out = []
        groups = (('meets', self.dec['feasible'] & self.dec['meets'], QColor(STATUS['good'])),
                  ('fails', self.dec['feasible'] & ~self.dec['meets'], QColor(STATUS['critical'])),
                  ('unreachable', ~self.dec['feasible'], THEME.color('ink2')))
        r = 4.0
        for text, mask, col in groups:
            sel = ok & mask[idx]
            polys = []
            for u, v in uv[sel]:
                polys.append(np.array([[u - r, v - r], [u + r, v + r]]))
                polys.append(np.array([[u - r, v + r], [u + r, v - r]]))
            if polys:
                out.append((polys, col, f'grasps: {text}'))
        return out

    # ---------------------------------------------------------------- explain
    def explain_text(self) -> str:
        st = self.settings()
        return (
            'For every grasp of the set the arm is solved by inverse kinematics (damped least squares from several '
            'seeds); a solution must keep inside the joint limits and clear of the robot itself and of the tables and '
            'objects. Unreachable grasps are counted by reason.\n\n'
            'At each solution, every camera is asked which points of each fingertip\'s contact patch it sees. The '
            f'point tested is {st["lift_mm"]:g} mm in front of the pad (where the object meets it, a default): it '
            'must fall inside the image (the lens model of the library entry) and nothing of the posed robot, the '
            'objects or the tables may lie in front of it (a cube map of distances around the camera, drawn on the '
            'GPU). A camera behind a finger has that finger in the way; a camera looking at the pinch from the side '
            'sees both fingertips.\n\n'
            f'The score is the share of the reachable grasps where {GV.requirement_text(st)} (your requirement; the '
            'default is every fingertip at 50 % in one camera), with a bootstrap interval. It reads good from '
            f'{st["good_pct"]:g} %, warning from {st["warn_pct"]:g} % and serious from {st["serious_pct"]:g} % '
            '(defaults 90, 70 and 40 %, not standards).\n\n'
            f'Pre-contact: the gripper backed off {st["precontact_cm"]:g} cm along its approach. What to move: each '
            f'camera moved by {100 * st["sens_step_m"]:g} cm along each axis or turned by {st["sens_step_deg"]:g}° '
            'about each of its axes, one at a time, on a subset of the grasps.\n\n'
            'Known limits: visibility is per cube-map texel; the fingers stay at their home opening; the contact patch '
            'comes from the collision geometry; IK is a local method; obstacles are checked at contact only; the '
            'moves are one camera and one axis at a time.')

    def explain(self):
        box = QMessageBox(self)
        box.setWindowTitle('How the Contact Visibility Score is computed')
        box.setText(self.explain_text())
        box.setModal(False)
        box.show()


def gallery_pixmap(view: dict, width: int) -> QPixmap:
    """A gallery view's render with its contact points drawn by state, scaled to width."""
    v = GV.decode_view(view)
    rgb = np.ascontiguousarray(v['rgb'], np.uint8)
    h, w = rgb.shape[:2]
    img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    for (u, vv), s in zip(v['uv'], v['state']):
        if not (-1 <= u <= w and -1 <= vv <= h):
            continue
        p.setBrush(QColor(STATE_COLORS.get(int(s), '#8a8a85')))
        p.drawEllipse(QPointF(float(u), float(vv)), 1.6, 1.6)
    p.end()
    pm = QPixmap.fromImage(img)
    return pm.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation) if w != width else pm


def _columns(tree: QTreeWidget, stretch: int = 0):
    h = tree.header()
    h.setStretchLastSection(False)
    for k in range(tree.columnCount()):
        h.setSectionResizeMode(k, QHeaderView.ResizeMode.Stretch if k == stretch else
                               QHeaderView.ResizeMode.ResizeToContents)


def _fit(tree: QTreeWidget):
    n = tree.topLevelItemCount()
    rh = tree.sizeHintForRow(0) if n else 20
    tree.setFixedHeight(tree.header().sizeHint().height() + max(1, n) * max(rh, 18) + 6)


register('grasp', 'Grasp self-occlusion', GraspPage, menu='Analysis', icon='grasp', action_id='grasp_study',
         workspace='grasp', order=30)
