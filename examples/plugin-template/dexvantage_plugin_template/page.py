"""The page of the template study (imported only when DexVantage's window is built): the worst aim angle as the
headline with its status word, a row per camera, Apply (one undo step) to aim the worst static camera at the task
volume's centre, Preview, Explain and Export (the base page's, with provenance), a 3D overlay (a line from each camera
to the centre, in its status colour) and a Camera view overlay (crosses at the volume's centre and the image centre).

Known limits: see the study's (aim.py).
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem

from dexvantage.ui.pages.base import StudyPage
from dexvantage.ui.theme import STATUS
from dexvantage.ui.widgets import label

from . import aim


class AimPage(StudyPage):
    """Aim at the task volume: the angle between each camera's viewing direction and its direction to the centre of
    the first task volume, at the home pose. The figure is the worst angle; at most the threshold (template default
    20°) reads "aimed"."""
    STUDY = aim.STUDY_ID
    TITLE = aim.TITLE
    HEADLINE = 'Worst camera aim off the task volume'

    def build_body(self, layout):
        self.advice = label('', wrap=True)
        layout.addWidget(self.advice)
        self.table = QTreeWidget()
        self.table.setRootIsDecorated(False)
        self.table.setHeaderLabels(['Camera', 'Angle', 'Distance', 'Mount', 'Aimed'])
        layout.addWidget(self.table)
        self._preview = False

    def show_result(self, doc):
        s = doc['summary']
        self.head.set(f'{s["worst_deg"]:.1f}', '°', s['status'], s['word'], s['advice'],
                      f'Threshold {doc["settings"]["max_angle_deg"]:.0f}° (template default)')
        self.advice.setText(s['advice'])
        self.table.clear()
        for r in doc['cameras']:
            QTreeWidgetItem(self.table, [r['name'], f'{r["angle_deg"]:.1f} °', f'{r["distance_m"]:.3f} m',
                                         'on a link' if r['on_link'] else 'static', 'yes' if r['aimed'] else 'no'])
        self.suggestion_ready(bool(doc.get('suggestion')))

    def apply(self):
        sg = (self.doc or {}).get('suggestion')
        if not sg:
            return
        self._preview = False
        self.ctx.store.edit('variant', lambda p: aim.apply_suggestion(p, sg), sg['text'])
        self.run()

    def preview(self):
        self._preview = not self._preview
        if self.ctx.window is not None:
            self.ctx.window.viewport.update()
            sg = (self.doc or {}).get('suggestion') or {}
            self.ctx.window.msg(f'Preview: {sg.get("text", "")} (the dashed line)' if self._preview else '', 5000)

    def overlays(self, vp) -> list:
        from dexvantage.ui.viewport3d import LineDraw
        if not self.doc:
            return []
        c = np.asarray(self.doc['volume']['center'], float)
        out = []
        sg = self.doc.get('suggestion') or {}
        for r in self.doc['cameras']:
            sev = 'good' if r['aimed'] else 'critical'
            o = np.asarray(r['origin'], float)
            dash = self._preview and r['sensor_uid'] == sg.get('sensor_uid')
            out.append(LineDraw(np.array([o, c], np.float32), QColor(STATUS[sev]), 2.0, dash=dash, depth=False))
        return out

    def camview_overlays(self, cam_model, pose) -> list:
        """The Camera view: a cross at the task volume's centre and a small one at the image centre, so the gap
        between them shows the aim error in pixels (the image centre stands in for the principal point)."""
        from dexvantage.core.camrender import project_world
        if not self.doc:
            return []
        uv, ok, _d = project_world(cam_model, pose, np.asarray(self.doc['volume']['center'], float)[None])
        W, H = cam_model.width, cam_model.height
        k = 0.03 * min(W, H)
        mid = np.array([(W - 1) / 2, (H - 1) / 2])
        out = [([mid + np.array([[-k / 2, 0], [k / 2, 0]]), mid + np.array([[0, -k / 2], [0, k / 2]])],
                QColor(STATUS['warning']), 'image centre')]
        if bool(ok[0]):
            u = uv[0]
            out.append(([u + np.array([[-k, -k], [k, k]]), u + np.array([[-k, k], [k, -k]])],
                        QColor(STATUS['good']), 'task volume centre'))
        return out

