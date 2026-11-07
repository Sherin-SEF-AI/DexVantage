"""The Plan against recording page (spec §8.3): a recorded dataset against the project's plan, camera by camera
(core/studies/plan_vs_recording.py).

What it shows: the share of comparisons that match the plan (the headline, with a status word: matches the plan when
all do, mostly matches from 80 %, differs from 50 %; DexVantage defaults), a row per recorded camera (the sensor it is
matched to and how, resolution, frame rate, mount, sync, visibility against the grasp study's prediction), the planned
cameras the recording lacks, and the visibility confusion of the selected camera. The dataset is the audit page's
(Data ▸ Audit a dataset…). Apply sets the plan's stream of each camera recorded at another size with the same aspect
ratio to the recorded size (one undo step), so the studies model what was recorded; Preview says what it would change.
The 3D view draws each camera's viewing direction in the colour of its worst comparison; the Camera view draws the
recorded image's border scaled into the planned image when the sizes differ.

Known limits: see the study's.
"""
from __future__ import annotations

import traceback

import numpy as np
from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHeaderView, QMessageBox, QTreeWidget, QTreeWidgetItem

from ...core.datasets.episode import format_title
from ...core.studies import plan_vs_recording as PR
from ..theme import STATUS
from ..widgets import label
from . import register
from .base import StudyPage

SEV = {'same': 'good', 'scaled': 'warning', 'differs': 'critical', 'matches': 'good', 'steady': 'good',
       'moved': 'serious', 'in sync': 'good', 'out of sync': 'critical', 'not measured': 'warning',
       'not stated by the plan': 'warning', 'yes': 'good', 'not in the plan': 'critical'}


class _Sig(QObject):
    fp = pyqtSignal(object)


def _f(x, nd=1):
    return '—' if x is None else f'{x:.{nd}f}'


class PlanVsRecordingPage(StudyPage):
    """Plan against recording: each recorded camera matched to the plan's sensor of the same name, then its
    resolution, frame rate, mount, sync and gripper visibility compared with what the plan says. The figure is the share
    of those comparisons that match."""
    STUDY = PR.SID
    TITLE = 'Plan against recording'
    HEADLINE = 'Comparisons that match the plan'

    def build_body(self, layout):
        self._sig = _Sig()
        self._sig.fp.connect(lambda fp: self.ctx.analysis.run(PR.SID, {'fingerprint': fp}))
        self.ds_label = label('', muted=True, small=True, wrap=True)
        layout.addWidget(self.ds_label)
        layout.addWidget(label('Cameras (click one for its visibility)', bold=True))
        self.cams = QTreeWidget()
        self.cams.setRootIsDecorated(False)
        self.cams.setHeaderLabels(['Camera', 'Plan sensor', 'Resolution', 'Frame rate', 'Mount', 'Sync',
                                   'Visibility'])
        h = self.cams.header()
        h.setStretchLastSection(False)
        for k in range(self.cams.columnCount()):
            h.setSectionResizeMode(k, QHeaderView.ResizeMode.ResizeToContents)
        self.cams.itemClicked.connect(lambda it, _c: self.show_camera(it.data(0, Qt.ItemDataRole.UserRole)))
        layout.addWidget(self.cams)
        self.missing_label = label('', wrap=True)
        layout.addWidget(self.missing_label)
        self.vis = QTreeWidget()
        self.vis.setRootIsDecorated(False)
        self.vis.setHeaderLabels(['', 'Recorded: gripper seen', 'Recorded: not seen'])
        self.vis.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)   # whole header words
        self.vis.header().setStretchLastSection(False)
        layout.addWidget(self.vis)
        self.vis_note = label('', muted=True, small=True, wrap=True)
        layout.addWidget(self.vis_note)
        self.ctx.store.project_replaced.connect(self._replaced)

    def run(self):
        from ...core.studies import audit as AU
        try:
            path = AU.dataset_path_for(self.ctx.store.project, PR.SID)
        except Exception as e:                          # noqa: BLE001 — no dataset: say so in the headline
            self.head.set('—', '', 'warning', 'no dataset', str(e))
            return
        import threading
        threading.Thread(target=lambda: self._sig.fp.emit(AU.fingerprint(path)), daemon=True).start()

    def _replaced(self):
        self.doc = None
        self.cams.clear()
        self.vis.clear()
        self.suggestion_ready(False)
        self.empty()

    def show_result(self, doc):
        try:
            s = doc.get('summary') or {}
            word = s.get('word', '')
            self.head.set(_f(s.get('match_pct'), 0) if s.get('match_pct') is not None else '—', '%',
                          s.get('status', 'warning'), word, s.get('advice', ''),
                          f'{s.get("match", 0)} of {s.get("checks", 0)} comparisons match (present, resolution, frame '
                          'rate, mount, sync, visibility per camera). Status: matches from 100 %, mostly from 80 %, '
                          'differs from 50 % (DexVantage defaults).')
            ds = doc.get('dataset') or {}
            self.ds_label.setText(f'{ds.get("name", "")}: {format_title(ds.get("format", ""), ds.get("version", ""))}, '
                                  f'{ds.get("episodes_audited", 0)} episodes audited.')
            self.cams.clear()
            for i, c in enumerate(doc.get('cameras') or []):
                r, f, m, sy = c.get('resolution') or {}, c.get('fps') or {}, c.get('mount') or {}, c.get('sync') or {}
                v = c.get('visibility') or {}
                cells = [c['key'].replace('observation.images.', ''), c.get('sensor') or c.get('present', ''),
                         f'{r.get("recorded", ["?", "?"])[0]} × {r.get("recorded", ["?", "?"])[1]} ({r.get("verdict")})'
                         if r else '—',
                         f'{_f(f.get("recorded"), 2)} fps ({f.get("verdict")})' if f else '—',
                         self._mount_text(m), f'{_f(sy.get("median_ms"))} ms ({sy.get("verdict")})' if sy else '—',
                         f'{100 * v["agreement"]:.0f} % agree' if v.get('frames') else '—']
                it = QTreeWidgetItem(cells)
                for col, verdict in ((2, r.get('verdict')), (3, f.get('verdict')), (4, m.get('verdict')),
                                     (5, sy.get('verdict'))):
                    if verdict in SEV:
                        it.setForeground(col, QColor(STATUS[SEV[verdict]]))
                it.setToolTip(1, f'Matched {c.get("matched") or ""}')
                pl = sy.get('plan') or {}
                if pl.get('predicted_offset_ms'):
                    lo, hi = pl['predicted_offset_ms']
                    it.setToolTip(5, f'Against the limit: {sy.get("verdict")}. Against the plan (the time-sync '
                                     f'study\'s skew of this stream): predicted offset {lo:.2f} to {hi:.2f} ms'
                                     + ('' if pl.get('complete') else ' (a lower bound: a term is not stated)')
                                     + f'; recorded {pl["verdict"]}.')
                elif pl.get('why'):
                    it.setToolTip(5, pl['why'])
                it.setData(0, Qt.ItemDataRole.UserRole, i)
                self.cams.addTopLevelItem(it)
            miss = doc.get('missing') or []
            self.missing_label.setText(('Planned but not recorded: ' + ', '.join(miss) + '.') if miss else
                                       'Every planned camera is in the recording.')
            if doc.get('cameras'):
                self.show_camera(0)
        except (KeyError, TypeError, ValueError):
            traceback.print_exc()
        self.suggestion_ready(bool(self.suggestion()))
        self._redraw()

    @staticmethod
    def _mount_text(m: dict) -> str:
        if not m or m.get('verdict') == 'not measured':
            return 'not measured'
        if m.get('source') == 'stored extrinsics':
            return f'{_f(m.get("position_mm"))} mm, {_f(m.get("rotation_deg"), 2)}° from the plan ({m["verdict"]})'
        return f'{_f(m.get("change_norm_deg"), 2)}° change over the campaign ({m.get("verdict")}, {m.get("source")})'

    def show_camera(self, i):
        rows = (self.doc or {}).get('cameras') or []
        if i is None or not 0 <= int(i) < len(rows):
            return
        v = rows[int(i)].get('visibility') or {}
        self.vis.clear()
        if not v.get('frames'):
            self.vis_note.setText('No visibility comparison for this camera.')
            return
        self.vis.addTopLevelItem(QTreeWidgetItem(['Predicted visible', str(v['both_seen']),
                                                  str(v['predicted_only'])]))
        self.vis.addTopLevelItem(QTreeWidgetItem(['Predicted hidden', str(v['recorded_only']),
                                                  str(v['both_hidden'])]))
        self.vis_note.setText(f'{v["frames"]} frames; agreement {100 * v["agreement"]:.0f} %. Predicted: the grasp '
                              'study\'s contact points at the recorded joint state; recorded: the gripper seen in the '
                              'frame (wrist: its correlation with its own appearance; static: its pixels differ from '
                              'the background).')

    # ---------------------------------------------------------------- suggestion
    def suggestion(self) -> list:
        """[(sensor uid, [w, h])] cameras recorded at another size of the same aspect ratio than their plan stream."""
        out = []
        for c in (self.doc or {}).get('cameras') or []:
            r = c.get('resolution') or {}
            if r.get('verdict') == 'scaled' and c.get('sensor_uid'):
                out.append((c['sensor_uid'], list(r['recorded'])))
        return out

    def apply(self):
        sg = self.suggestion()
        if not sg:
            return None

        def fn(p):
            for uid, wh in sg:
                s = p.variant.sensor(uid)
                if s is not None:
                    s.stream = [int(wh[0]), int(wh[1])]
        n = len(sg)
        self.ctx.store.edit('variant', fn, f'Plan against recording · {n} camera stream{"" if n == 1 else "s"} set to '
                                           'the recording')
        self.suggestion_ready(False)
        return sg

    def preview(self):
        sg = self.suggestion()
        if self.ctx.window is not None:
            v = self.ctx.store.project.variant

            def name(uid):
                s = v.sensor(uid)
                return s.name if s is not None else uid
            self.ctx.window.msg('Apply would set the stream of ' + '; '.join(f'{name(uid)} to {w} × {h} px'
                                                                          for uid, (w, h) in sg) +
                                ' (the size recorded, same aspect ratio).' if sg else 'Nothing to change.', 8000)

    def explain(self):
        box = QMessageBox(self)
        box.setWindowTitle('How a recording is compared with the plan')
        box.setText('Each recorded camera is matched to the plan\'s sensor of the same name. Resolution: the recorded '
                    'size against the plan\'s stream (same, scaled with the same aspect ratio, or different). Frame '
                    'rate: measured from the timestamps against the plan\'s stream. Mount: stored extrinsics against '
                    'the plan, else the audit\'s estimates over the campaign. Sync: the audit\'s camera offsets. '
                    'Visibility: the grasp study\'s prediction at the recorded joint states against the gripper seen '
                    'in the frames. Every tolerance is a DexVantage default.')
        box.setModal(False)
        box.show()

    # ---------------------------------------------------------------- overlays
    def overlays(self, vp) -> list:
        from ...core import camrender as CR
        from ...core import library as LB
        from ...core.frames import quat_to_matrix
        from ...core.studies import coverage as CV
        from ..viewport3d import LineDraw
        out = []
        p = self.ctx.store.project
        rank = {'good': 0, 'warning': 1, 'serious': 2, 'critical': 3}
        for c in (self.doc or {}).get('cameras') or []:
            s = p.variant.sensor(c.get('sensor_uid') or '')
            if s is None:
                continue
            verdicts = [(c.get(k) or {}).get('verdict') for k in ('resolution', 'fps', 'mount', 'sync')]
            sev = max((SEV.get(v, 'good') for v in verdicts if v), key=lambda x: rank[x], default='good')
            try:
                im, _w = CV.pick_imager(s, LB.sensor_imagers(s, CV.sensor_mode(s, {})), {})
                pose = CR.optical_pose(p, s, im, self.ctx.store.q)
            except Exception:                           # noqa: BLE001
                continue
            o = np.asarray(pose.position, float)
            z = quat_to_matrix(pose.quaternion)[:, 2]
            out.append(LineDraw(np.array([o, o + 0.15 * z]), QColor(STATUS[sev]), width=3.0, depth=False))
        return out

    def camview_overlays(self, cam_model, pose) -> list:
        """The recorded image's border inside the planned image when the recording's aspect ratio differs."""
        cv = getattr(self.ctx.window, 'camview', None) if self.ctx.window is not None else None
        uid = getattr(cv, 'sensor_uid', None) or getattr(getattr(cv, 'view', None), 'sensor_uid', None)
        for c in (self.doc or {}).get('cameras') or []:
            r = c.get('resolution') or {}
            if c.get('sensor_uid') != uid or r.get('verdict') != 'differs':
                continue
            W, H = cam_model.width, cam_model.height
            w, h = r['recorded']
            s = min(W / w, H / h)                       # the recorded frame fitted into the planned one, centred
            x0, y0 = (W - w * s) / 2 - 0.5, (H - h * s) / 2 - 0.5
            box = np.array([[x0, y0], [x0 + w * s, y0], [x0 + w * s, y0 + h * s], [x0, y0 + h * s], [x0, y0]])
            return [([box], QColor(STATUS['critical']), f'recorded {w} × {h} (aspect ratio differs from the plan)')]
        return []

    def _redraw(self):
        win = self.ctx.window
        if win is not None and hasattr(win, 'viewport'):
            win.viewport.update()
        cv = getattr(win, 'camview', None) if win is not None else None
        if cv is not None and hasattr(cv, 'view'):
            cv.view.refresh_overlays()


register('plan_vs_recording', 'Plan against recording', PlanVsRecordingPage, menu='Data', icon='compare',
         action_id='plan_vs_recording', workspace='audit', order=81)
