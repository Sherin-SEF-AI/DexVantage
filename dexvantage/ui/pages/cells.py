"""The Cells page (File ▸ Compare cells…): the open project against other robot cells (core/studies/cells.py,
model in core/cells.py and core/cells_pool.py).

What it shows: the headline (the largest geometric gap index to the reference cell, 0 identical … 1 unrelated, with
its band as the status word: small, moderate, large; a heuristic), the cells (the open project first, its other
variants, files and samples added here), the gap per cell with its largest component and the matched cameras of the
chosen cell (viewpoint angle, distance, FOV, resolution ratio, task volume in the image), the figures of merit side by
side (best green, worst red; with tolerance draws, whether each difference exceeds the build spread), the pooling
verdict and mapping per cell, and the drift of the design across the cells (measured from hand-eye calibrations, or
as designed). Every setting is one undo step. Apply makes the suggestion (one undo step): mount the open project's
least-alike camera at the reference's pose, or add an as-built variant at the copies' median; Preview draws it.
Export writes cells.json, cells_pooling.json and cells_comparison.html into a folder, with provenance.

Known limits: see the study's.
"""
from __future__ import annotations

import os
import traceback

import numpy as np
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QCheckBox, QComboBox, QFileDialog, QHeaderView, QMessageBox, QTreeWidget, QTreeWidgetItem

from ...core import cells as CE
from ..theme import STATUS
from ..widgets import button, label
from . import register
from .base import StudyPage, row

DRAWS = (0, 8, 16, 32, 64)
VERDICT_SEV = {'poolable as is': 'good', 'poolable after mapping': 'warning', 'not poolable': 'serious'}


def _f(v, nd=3, unit=''):
    if v is None:
        return '—'
    return f'{v:.{nd}f}{(" " + unit) if unit else ""}'


def _tree(headers, height=None):
    t = QTreeWidget()
    t.setRootIsDecorated(False)
    t.setHeaderLabels(headers)
    t.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    if height:
        t.setMaximumHeight(height)
    return t


def fingerprint(entries) -> list:
    """[(name, size, modification time)] of the cells' files, so a file changed on disk misses the result cache."""
    out = []
    for e in entries or []:
        p = e.get('path') if isinstance(e, dict) else (e if isinstance(e, str) and not e.startswith('sample:') else
                                                      None)
        if not p:
            continue
        try:
            st = os.stat(p)
            out.append([os.path.basename(os.path.normpath(p)), int(st.st_size), int(st.st_mtime_ns)])
        except OSError:
            out.append([os.path.basename(os.path.normpath(p)), None, None])
    return out


class CellsPage(StudyPage):
    """Cells: the open project against other robot cells (other labs' or benches' rigs, samples, or this project's
    variants): the figures of merit side by side, the geometric gap a policy meets from one cell's cameras to
    another's (per camera role), whether their recordings pool for training and with what mapping, and how far the
    built copies of one design have drifted from it."""
    STUDY = CE.SID
    TITLE = 'Cells'
    HEADLINE = 'Largest geometric gap to the reference cell'
    MENU_TITLE = 'Compare cells…'
    MENU_AFTER = 'compare'

    def build_body(self, layout):
        self._preview = False
        self._filling = False
        self.cell_tab = _tree(['#', 'Cell', 'Source', 'Robots', 'Cameras'], 150)
        layout.addWidget(label('Cells', bold=True))
        layout.addWidget(self.cell_tab)
        self.b_add = button('Add cells…', role='ghost', icon_name='folder', tip='Other projects (.dexv): other labs\' '
                            'or benches\' cells')
        self.b_add.clicked.connect(lambda: self.add_files())
        self.sample_box = QComboBox()
        self.sample_box.setToolTip('Add a sample project as a cell')
        self.sample_box.activated.connect(self._sample_picked)
        self.b_remove = button('Remove', role='ghost', tip='Remove the selected cell (the open project stays)')
        self.b_remove.clicked.connect(self.remove_selected)
        self.c_variants = QCheckBox('This project\'s variants are cells')
        self.c_variants.setToolTip('Each other variant of the open project is a cell too (default on)')
        self.c_variants.toggled.connect(lambda on: self._set('variants', bool(on), 'variants as cells'))
        layout.addWidget(row(self.b_add, self.sample_box, self.b_remove, self.c_variants))
        self.ref_box = QComboBox()
        self.ref_box.setToolTip('The cell the others are compared with (the plan, for the drift)')
        self.ref_box.activated.connect(lambda i: self._set('reference', int(i), 'reference cell'))
        self.met_box = QComboBox()
        self.met_box.addItems(['Quick figures (default)', 'Every figure (slow)'])
        self.met_box.setToolTip('Which figures of merit to compare: the quick ones (coverage, depth, sync, bandwidth, '
                                'the rig), or every study\'s')
        self.met_box.activated.connect(lambda i: self._set('metrics', None if i == 0 else 'all', 'figures'))
        self.draw_box = QComboBox()
        self.draw_box.addItems(['Off (default)'] + [f'{n} draws' for n in DRAWS[1:]])
        self.draw_box.setToolTip('Tolerance Monte Carlo per cell: whether a difference is more than the build '
                                 'tolerances spread it (each draw runs every study named)')
        self.draw_box.activated.connect(lambda i: self._set('tolerance_draws', DRAWS[i], 'tolerance draws'))
        layout.addWidget(row(label('Reference', ink2=True), self.ref_box, label('Figures', ink2=True), self.met_box,
                             label('Tolerance spread', ink2=True), self.draw_box))
        layout.addWidget(label('Geometric gap to the reference', bold=True))
        self.gap_tab = _tree(['Cell', 'Gap index', 'Band', 'Largest component', 'Cameras matched'], 140)
        self.gap_tab.currentItemChanged.connect(lambda *_: self._show_matches())
        layout.addWidget(self.gap_tab)
        self.match_tab = _tree(['Camera', 'Matched with', 'Role', 'Viewpoint', 'Rotation', 'Distance', 'HFOV Δ',
                                'Resolution', 'Task volume in image'], 120)
        layout.addWidget(self.match_tab)
        self.findings = label('', muted=True, wrap=True, small=True)
        layout.addWidget(self.findings)
        layout.addWidget(label('Figures of merit', bold=True))
        self.met_tab = _tree(['Figure'], 220)
        layout.addWidget(self.met_tab)
        self.met_note = label('', muted=True, wrap=True, small=True)
        layout.addWidget(self.met_note)
        layout.addWidget(label('Dataset pooling', bold=True))
        self.pool_tab = _tree(['Cell', 'Verdict', 'Compared', 'Mapping steps', 'Cannot be mapped'], 140)
        layout.addWidget(self.pool_tab)
        self.pool_note = label('', muted=True, wrap=True, small=True)
        layout.addWidget(self.pool_note)
        layout.addWidget(label('Drift of the design across the cells', bold=True))
        self.drift_tab = _tree(['Camera', 'Cell', 'Kind', 'Offset', 'Rotation', 'Share of tolerance', 'Verdict',
                                'Source'], 140)
        layout.addWidget(self.drift_tab)
        self.drift_note = label('', muted=True, wrap=True, small=True)
        layout.addWidget(self.drift_note)
        self.notes = label('', muted=True, wrap=True, small=True)
        layout.addWidget(self.notes)
        self.ctx.store.changed.connect(self._changed)
        self.ctx.store.project_replaced.connect(self._replaced)
        self._fill_controls()

    # ---------------------------------------------------------------- settings (each one undo step)
    def settings(self) -> dict:
        return dict(self.ctx.store.project.analysis.studies.get(CE.SID) or {})

    def _set(self, key, value, what):
        if self._filling:
            return
        st = self.settings()
        if st.get(key) == value and key in st:
            return

        def fn(p):
            s = dict(p.analysis.studies.get(CE.SID) or {})
            if value is None:
                s.pop(key, None)
            else:
                s[key] = value
            p.analysis.studies[CE.SID] = s
        self.ctx.store.edit('analysis', fn, f'Cells · {what}')

    def _set_cells(self, entries, what):
        self._set('cells', list(entries), what)

    def add_files(self, paths=None):
        if paths is None:
            paths, _ = QFileDialog.getOpenFileNames(self, 'Add cells', '',
                                                    'DexVantage projects (*.dexv);;All files (*)')
        paths = [p for p in paths or [] if p]
        gone = [p for p in paths if not os.path.exists(p)]
        if gone:                                         # said now, not as the whole comparison's refusal later
            self._say(f'There is no file at {os.path.basename(gone[0].rstrip(os.sep))}: it was not added as a cell.')
            paths = [p for p in paths if p not in gone]
        if not paths:
            return
        cur = list(self.settings().get('cells') or [])
        have = {e.get('path') for e in cur if isinstance(e, dict)}
        cur += [{'path': p} for p in paths if p not in have]
        self._set_cells(cur, f'add {len(paths)} cell{"" if len(paths) == 1 else "s"}')

    def add_sample(self, name):
        cur = list(self.settings().get('cells') or [])
        cur.append({'sample': name})
        self._set_cells(cur, f'add the {name} sample')

    def _sample_picked(self, i):
        if self._filling or i <= 0:
            return
        name = self.sample_box.itemData(i)
        self.sample_box.setCurrentIndex(0)
        if name:
            self.add_sample(name)

    def remove_selected(self):
        it = self.cell_tab.currentItem()
        if it is None:
            return
        k = it.data(0, 256)
        if not isinstance(k, int) or k < 0:
            self._say('The open project and its variants are always cells (turn the variants off above).')
            return
        cur = list(self.settings().get('cells') or [])
        if 0 <= k < len(cur):
            del cur[k]
            self._set_cells(cur, 'remove a cell')

    def _fill_controls(self):
        """The cell list, the reference choices and the settings as stored (no study run: cheap)."""
        self._filling = True
        try:
            p = self.ctx.store.project
            st = self.settings()
            self.c_variants.setChecked(bool(st.get('variants', True)))
            self.met_box.setCurrentIndex(1 if st.get('metrics') == 'all' else 0)
            d = int(st.get('tolerance_draws') or 0)
            self.draw_box.setCurrentIndex(DRAWS.index(d) if d in DRAWS else 0)
            if self.sample_box.count() == 0:
                from ...core.robot.samples import list_samples
                self.sample_box.addItem('Add a sample…', None)
                for s in list_samples():
                    if s.get('available', True):
                        self.sample_box.addItem(s['title'], s['name'])
            names = [(f'{p.name} (open project)', 'open project', -1)]
            if st.get('variants', True):
                names += [(f'{p.name} · {v.name}', 'variant', -1) for v in p.variants if v.uid != p.active_uid]
            for k, e in enumerate(st.get('cells') or []):
                if isinstance(e, str):
                    e = {'sample': e[7:]} if e.startswith('sample:') else {'path': e}
                if not isinstance(e, dict):
                    continue
                if e.get('sample'):
                    names.append((f'sample {e["sample"]}', 'sample', k))
                elif e.get('path'):
                    names.append((os.path.basename(os.path.normpath(e['path'])), 'file', k))
                elif e.get('variant'):
                    names.append((f'variant {e["variant"]}', 'variant', k))
            res = {c['index']: c for c in (self.doc or {}).get('cells') or []}
            self.cell_tab.clear()
            for i, (n, src, k) in enumerate(names):
                c = res.get(i) if self.doc is not None and len(res) == len(names) else None
                it = QTreeWidgetItem([str(i), c['name'] if c else n, src, ', '.join(c['robots']) if c else '',
                                      str(c['cameras']) if c else ''])
                it.setData(0, 256, k)
                self.cell_tab.addTopLevelItem(it)
            self.ref_box.clear()
            self.ref_box.addItems([f'{i}: {n}' for i, (n, _s, _k) in enumerate(names)])
            ref = int(st.get('reference', 0) or 0)
            self.ref_box.setCurrentIndex(ref if 0 <= ref < len(names) else 0)
            self.b_run.setEnabled(len(names) >= 2)
            if len(names) < 2 and self.doc is None:
                self.head.set('—', '', 'warning', 'one cell', 'Add another cell to compare with: another project '
                              '(Add cells…), a sample, or a variant of this project.')
        finally:
            self._filling = False

    def _changed(self, sections):
        if 'analysis' in sections or 'variants' in sections or 'meta' in sections:
            self._fill_controls()

    def _replaced(self):
        self.doc = None
        self._preview = False
        self.suggestion_ready(False)
        self.empty()
        self._fill_controls()

    def activate(self):
        self._fill_controls()

    def run(self):
        st = self.settings()
        self.ctx.analysis.run(self.STUDY, settings={'_fingerprint': fingerprint(st.get('cells'))})

    def _say(self, text, ms=6000):
        if self.ctx.window is not None:
            self.ctx.window.msg(text, ms)

    # ---------------------------------------------------------------- the result
    def show_result(self, doc):
        try:
            h = doc['headline']
            self.head.set(f'{h["value"]:.3f}', '', h['status'], h['word'], h['advice'],
                          f'{h["text"]} The gap index is 0 for identical cells and 1 for unrelated ones; the bands '
                          f'(< 0.15 small, < 0.35 moderate, else large) are a heuristic, not a predicted success '
                          f'rate.')
            self._fill_controls()
            names = [c['name'] for c in doc['cells']]
            self.gap_tab.clear()
            for g in doc['gap']['pairs']:
                comp = g['components']
                big = max(comp, key=comp.get)
                it = QTreeWidgetItem([g['b'], f'{g["total"]:.3f}', g['band'], f'{big} {comp[big]:.2f}',
                                      f'{len(g["matches"])} of {g["cameras"][1]}'])
                it.setForeground(2, QColor(STATUS[g['status']]))
                self.gap_tab.addTopLevelItem(it)
            if self.gap_tab.topLevelItemCount():
                self.gap_tab.setCurrentItem(self.gap_tab.topLevelItem(0))
            self._show_matches()
            self._fill_metrics(doc, names)
            self._fill_pooling(doc)
            self._fill_drift(doc)
            self.notes.setText(' '.join(doc.get('notes') or []))
        except (KeyError, TypeError, ValueError):
            traceback.print_exc()
        sg = doc.get('suggestion')
        self.suggestion_ready(bool(sg) and not sg.get('locked'))
        self._redraw()

    def _show_matches(self):
        self.match_tab.clear()
        doc = self.doc or {}
        it = self.gap_tab.currentItem()
        pairs = (doc.get('gap') or {}).get('pairs') or []
        if it is None or not pairs:
            self.findings.setText('')
            return
        g = pairs[self.gap_tab.indexOfTopLevelItem(it)]
        for m in g['matches']:
            self.match_tab.addTopLevelItem(QTreeWidgetItem([
                m['a'], m['b'], m['role'], f'{m["viewpoint_deg"]:.2f}°', f'{m["rotation_deg"]:.2f}°',
                f'{m["distance_m"] * 1000:.1f} mm', f'{m["hfov_diff_deg"]:+.1f}°',
                f'{m["resolution_ratio"]:.2f}×' if m['resolution_ratio'] else '—',
                f'{_f(m["image_pct_a"], 0)} → {_f(m["image_pct_b"], 0)} %']))
        self.findings.setText(' '.join(g['findings'][1:]))

    def _fill_metrics(self, doc, names):
        mt = doc['metrics']
        self.met_tab.clear()
        self.met_tab.setColumnCount(1 + len(names))
        self.met_tab.setHeaderLabels(['Figure'] + names)
        for r in mt['rows']:
            vals = []
            for k, v in enumerate(r['values']):
                b = {'at_least': '≥ ', 'at_most': '≤ '}.get(r['bounds'][k] or '', '')
                vals.append('—' if v is None else f'{b}{v:.4g} {r["unit"]}')
            it = QTreeWidgetItem([r['title']] + vals)
            for k in range(len(names)):
                tip = r['reasons'][k] or ''
                d = next((x for x in r['differences'] if x['cell'] == k), None)
                if d and d['why']:
                    tip = (tip + ' ' if tip else '') + f'Against the reference: {d["why"]}.'
                if tip:
                    it.setToolTip(1 + k, tip)
                if k == r['best']:
                    it.setForeground(1 + k, QColor(STATUS['good']))
                elif k == r['worst']:
                    it.setForeground(1 + k, QColor(STATUS['serious']))
                if d and d['meaningful']:
                    it.setText(1 + k, it.text(1 + k) + ' *')
            self.met_tab.addTopLevelItem(it)
        tol = mt['tolerance']
        st = tol.get('stats') or {}
        self.met_note.setText('Green: best; red: worst (where the cells differ); — : not stated (hover for why). '
                              + (f'* more than the build spread ({tol["rule"]}'
                                 + (f'; and significant: {st["rule"]}' if st.get('rule') else '')
                                 + f'; {mt["meaningful"]} of {mt["assessed"]} differences).' if tol['draws']
                                 else tol['rule'] + '.'))

    def _fill_pooling(self, doc):
        p = doc['pooling']
        self.pool_tab.clear()
        for c in p['cells']:
            if c['target']:
                continue
            ops = ', '.join(sorted({o['op'] for o in c['operations']})) or 'none'
            bad = '; '.join(u['what'] for u in c['unmappable']) or '—'
            it = QTreeWidgetItem([c['name'], c['verdict'], 'recording' if c['source'] == 'recorded' else 'plan',
                                  f'{len(c["operations"])} ({ops})', bad])
            it.setForeground(1, QColor(STATUS[VERDICT_SEV[c['verdict']]]))
            it.setToolTip(4, '\n'.join(u['why'] for u in c['unmappable']))
            it.setToolTip(3, '\n'.join(f'{o["op"]} {o.get("feature", "")}: {o["why"]}' for o in c['operations']))
            self.pool_tab.addTopLevelItem(it)
        self.pool_note.setText(f'{p["text"]} Target: {p["target"]["name"]} ({p["target"]["what"]}). Export writes '
                               'the mapping as JSON.')

    def _fill_drift(self, doc):
        dr = doc['drift']
        self.drift_tab.clear()
        for r in dr['cameras']:
            for x in r['copies']:
                if x.get('kind') == 'not comparable':
                    self.drift_tab.addTopLevelItem(QTreeWidgetItem([r['sensor'], x['name'], 'not comparable', '', '',
                                                                    '', '', x['why']]))
                    continue
                it = QTreeWidgetItem([r['sensor'], x['name'], x['kind'], f'{x["offset_mm"]:.2f} mm',
                                      f'{x["rotation_deg"]:.3f}°', f'{x["ratio"]:.2f}', x['verdict'], x['source']])
                it.setForeground(6, QColor(STATUS[CE.VERDICT_SEV[x['verdict']]]))
                self.drift_tab.addTopLevelItem(it)
        tol = next((r for r in dr['cameras']), None)
        bits = []
        if tol is not None:
            bits.append(f'Tolerance ± {tol["tolerance_mm"]:g} mm, ± {tol["tolerance_deg"]:g}°'
                        + (' (the defaults)' if tol['tolerance_default'] else '') + '.')
        bits += [f'{c["sensor"]}: built consistently off the plan (the copies\' median at {c["median_ratio"]:.2f} of '
                 'the tolerance).' for c in dr['consistency'] if c['systematic']]
        bits += [f'{o["name"]}: {c["sensor"]} stands out from the other copies (robust z {o["z"]:+.1f}).'
                 for c in dr['consistency'] for o in c['outliers']]
        if not dr['measured_cells']:
            bits.append('No cell has a hand-eye calibration of these cameras: the rows compare designs, not builds.')
        self.drift_note.setText(' '.join(bits))

    # ---------------------------------------------------------------- the suggestion
    def apply(self):
        sg = (self.doc or {}).get('suggestion')
        if not sg:
            return None
        if sg.get('locked'):
            self._say(f'{sg["sensor"]} is locked: unlock it to move it.')
            return None
        from ...core.frames import Pose
        if sg['kind'] == 'match_mount':
            def fn(p):
                s = p.variant.sensor(sg['sensor_uid'])
                if s is not None and not s.locked:
                    s.mount.pose = Pose.from_dict(sg['pose'])
            self.ctx.store.edit('variant', fn, f'Cells · {sg["sensor"]} at {sg["reference"]}\'s pose '
                                               f'({sg["move_m"] * 1000:.0f} mm)')
        elif sg['kind'] == 'as_built_variant':
            def fn(p):
                from ...core.project import Variant, new_uid
                v = Variant.from_dict(p.variant.to_dict())
                v.uid, v.name = new_uid(), sg['name']
                v.note = 'Added by the Cells page: the cameras at the median of their calibrated copies.'
                for s in v.sensors:
                    old = s.uid
                    s.uid = new_uid()
                    if old in sg['poses']:
                        s.mount.pose = Pose.from_dict(sg['poses'][old])
                p.variants.append(v)
            self.ctx.store.edit('variants', fn, f'Cells · add the variant {sg["name"]}')
            self._say(f'Added the variant "{sg["name"]}" (the plan is kept and stays active).', 8000)
        else:
            return None
        self._preview = False
        self.suggestion_ready(False)
        self._redraw()
        return sg

    def preview(self):
        sg = (self.doc or {}).get('suggestion')
        self._preview = not self._preview and bool(sg)
        if sg:
            self._say(sg['text'] + (' The dashed lines in the 3D view show it.' if self._preview else ''), 10000)
        self._redraw()

    def explain(self):
        box = QMessageBox(self)
        box.setWindowTitle('How the cells are compared')
        box.setText('Cameras are matched within their role: wrist cameras (on a link a joint moves) in their robot\'s '
                    'contact frame, between the fingertips; scene cameras in the world frame. For each pair: the angle '
                    'between the optical axes, the distance between the centres, the FOV and pixels-per-degree '
                    'ratios, and how much of the image the task volume fills. The gap index is a weighted mean of '
                    'those (viewpoint 0.30, position 0.20, resolution 0.15, coverage 0.15, FOV 0.10, camera count '
                    '0.10), 0 for identical cells; the bands are a heuristic ported from Vantage. Pooling compares '
                    'the LeRobot features each cell records (or plans to): cameras, sizes, rates, codecs, joints, '
                    'action space, gripper encoding, control rate, and lists the mapping steps. Drift compares each '
                    'built copy\'s calibrated camera poses with the plan, as a share of the mount tolerance.')
        box.setModal(False)
        box.show()

    # ---------------------------------------------------------------- export: the three files, with provenance
    def export(self, path=None):
        if self.doc is None:
            if path is None:
                self._say('There is no cell comparison to export yet: Run it first.', 5000)
            return None
        folder = path
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the cell comparison into a folder')
            if not folder:
                return None
        from ...core.jobs import StudyError
        try:
            names = CE.write_files(self.doc, self.ctx.store.project, folder)
        except StudyError as e:
            self._say(str(e), 8000)
            return None
        self._say(f'Exported {", ".join(names)} to {os.path.basename(os.path.normpath(folder)) or folder}', 8000)
        return [os.path.join(folder, n) for n in names]

    # ---------------------------------------------------------------- overlays
    def overlays(self, vp) -> list:
        from ..viewport3d import LineDraw
        sg = (self.doc or {}).get('suggestion')
        if not self._preview or not sg:
            return []
        out = []
        try:
            p = self.ctx.store.project
            from ...core.scenegeom import sensor_world_pose
            if sg['kind'] == 'match_mount':
                T = np.asarray(sg['T_world_optical'], float)
                s = p.variant.sensor(sg['sensor_uid'])
                o, z = T[:3, 3], T[:3, 2]
                x, y = T[:3, 0], T[:3, 1]
                d = 0.08
                corners = [o + d * (z + 0.5 * (sx * x + sy * y)) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                for c in corners:
                    out.append(LineDraw(np.array([o, c]), QColor('#d95926'), width=1.5, dash=True, depth=False))
                out.append(LineDraw(np.array(corners + [corners[0]]), QColor('#d95926'), width=1.5, dash=True,
                                    depth=False))
                if s is not None:
                    out.append(LineDraw(np.array([sensor_world_pose(p, s).position, o]), QColor('#d95926'),
                                        width=1.0, dash=True, depth=False))
            elif sg['kind'] == 'as_built_variant':
                from ...core.frames import Pose
                for uid, pose in sg['poses'].items():
                    s = p.variant.sensor(uid)
                    if s is None:
                        continue
                    a = sensor_world_pose(p, s).position
                    q = type(s).from_dict(s.to_dict())
                    q.mount.pose = Pose.from_dict(pose)
                    b = sensor_world_pose(p, q).position
                    out.append(LineDraw(np.array([a, b]), QColor('#d95926'), width=2.0, dash=True, depth=False))
        except Exception:                               # noqa: BLE001 — an overlay must not take the view down
            traceback.print_exc()
            return []
        return out

    def _redraw(self):
        win = self.ctx.window
        if win is not None and hasattr(win, 'viewport'):
            win.viewport.update()


register('cells', 'Cells', CellsPage, menu='File', icon='compare', action_id=None, workspace=None, order=120)
