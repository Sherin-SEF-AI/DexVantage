"""The VLA token resolution page (spec §7.12): how many of each policy's ViT tokens the task objects and the
fingertips cover, per camera, at the pre-grasp and contact poses of a grasp set (core/studies/vla_tokens.py).

What it shows: the fewest tokens a critical object touches (the headline, with a status word: "too few tokens" when a
critical object spans fewer than 2 × 2 tokens, a default; "partly cropped" when the crop cuts part of it away), the
image-to-input mapping of the chosen policy and camera (the crop box in sensor pixels, the scale, the token grid and
where each figure comes from), a table of objects and one of fingertips per camera, and the warnings. Apply sets the
suggested camera stream (one undo step); Preview says what it would change; Export writes the result with provenance;
"Export Plücker rays…" writes the policy's ray maps (core/plucker.py: .npy, .safetensors and a provenance JSON per
camera) at the pose on screen.

Overlays, at the pose on screen: the Camera view draws the crop boundary, the token grid over the part of the image
the input keeps and the tokens covering each critical object; the 3D view draws those tokens' footprints on the
objects (each token's boundary cast from the camera onto the object).

Known limits: the overlays follow the pose on screen while the tables are for the poses counted (the grasp set's, or
the pose of the run); see the study's own known limits (no occlusion test, inference-time crop only).
"""
from __future__ import annotations

import json
import traceback

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QGridLayout, QHeaderView, QMessageBox, QTreeWidget,
                             QTreeWidgetItem, QWidget)

from ...core.studies import vla_tokens as VT
from ..theme import STATUS, THEME, mono
from ..widgets import ScrubSpinBox, button, label
from . import register
from .base import StudyPage, row

PHASE = {'pre_grasp': 'pre-grasp', 'contact': 'contact', 'home': 'home'}
MAX_OVERLAY_CELLS = 400


def _f(x, fmt='{:.1f}') -> str:
    return '—' if x is None else fmt.format(x)


def _pct(x) -> str:
    return '—' if x is None else f'{100 * x:.0f} %'


class VlaPage(StudyPage):
    """VLA token resolution: each policy resizes or crops a camera image to its input and slices it into ViT patches;
    an object covering fewer than 2 × 2 of them (a default) gives the policy little to see. The headline is the fewest
    tokens a critical object touches in a camera that sees it, over the policies, cameras and poses counted."""
    STUDY = 'vla_tokens'
    TITLE = 'VLA token resolution'
    HEADLINE = 'Fewest tokens on a critical object'

    # ---------------------------------------------------------------- body
    def build_body(self, layout):
        self._overlay3d = None
        self._ov_key = None
        self.pol_box = QComboBox()
        self.pol_box.setToolTip('The policy whose input, grid and counts are shown (and drawn)')
        self.cam_box = QComboBox()
        self.cam_box.setToolTip('The camera whose mapping and tables are shown')
        for b in (self.pol_box, self.cam_box):
            b.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
            b.setMinimumWidth(160)
        self.pol_box.currentIndexChanged.connect(lambda _i: self._selection_changed(True))
        self.cam_box.currentIndexChanged.connect(lambda _i: self._selection_changed(False))
        layout.addWidget(row(label('Policy'), self.pol_box, label('Camera'), self.cam_box))
        self.mapping = label('', mono_=True, wrap=True)
        self.mapping.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.mapping)
        layout.addWidget(label('Task objects', bold=True))
        self.objs = QTreeWidget()
        self.objs.setRootIsDecorated(False)
        self.objs.setHeaderLabels(['Object', 'Tokens (fewest)', 'Half covered', 'Extent (tokens)', 'Outside crop',
                                   'Poses in input', 'Status'])
        _columns(self.objs)
        layout.addWidget(self.objs)
        layout.addWidget(label('Fingertips', bold=True))
        self.tips = QTreeWidget()
        self.tips.setRootIsDecorated(False)
        self.tips.setHeaderLabels(['Fingertip', 'Pose', 'Tokens (fewest)', 'Median', 'Most', 'Pad points in input'])
        _columns(self.tips)
        layout.addWidget(self.tips)
        layout.addWidget(label('Warnings', bold=True))
        self.warn_label = label('', wrap=True)
        layout.addWidget(self.warn_label)
        self.sugg_label = label('', muted=True, wrap=True)
        layout.addWidget(self.sugg_label)
        self.notes_label = label('', muted=True, wrap=True)
        layout.addWidget(self.notes_label)

        layout.addWidget(label('Settings (yours; defaults labelled)', bold=True))
        g = QWidget()
        grid = QGridLayout(g)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        self.side_spin = ScrubSpinBox('', 'plain', 1, 'tokens', step=0.5, width=110)
        self.side_spin.setRange(0.5, 64.0)
        self.side_spin.setToolTip('A critical object spanning fewer tokens than this along either side warns '
                                  f'(default {VT.DEFAULTS["min_tokens_side"]:g})')
        self.side_def = label('', muted=True, small=True)
        self.pre_spin = ScrubSpinBox('', 'plain', 0, 'mm', step=10.0, width=110)
        self.pre_spin.setRange(0.0, 2000.0)
        self.pre_spin.setToolTip('The pre-grasp pose: the gripper backed off this far along its approach '
                                 f'(default {VT.DEFAULTS["pre_grasp_m"] * 1000:g} mm)')
        self.pre_def = label('', muted=True, small=True)
        grid.addWidget(label('Smallest span'), 0, 0)
        grid.addWidget(self.side_spin, 0, 1)
        grid.addWidget(self.side_def, 0, 2)
        grid.addWidget(label('Pre-grasp distance'), 1, 0)
        grid.addWidget(self.pre_spin, 1, 1)
        grid.addWidget(self.pre_def, 1, 2)
        self.gs_box = QComboBox()
        self.gs_box.setToolTip('The grasp set whose pre-grasp and contact poses are counted')
        self.gs_box.activated.connect(lambda _i: self._set_setting(
            'grasp_set', self.gs_box.currentData() or None, f'VLA tokens · grasp set {self.gs_box.currentText()}'))
        grid.addWidget(label('Grasp set'), 2, 0)
        grid.addWidget(self.gs_box, 2, 1, 1, 2)
        layout.addWidget(g)
        self.crit_box = QWidget()
        self.crit_layout = QGridLayout(self.crit_box)
        self.crit_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(label('Critical objects (every object by default)', muted=True))
        layout.addWidget(self.crit_box)
        self.side_spin.committed.connect(lambda v: self._set_setting(
            'min_tokens_side', float(v), f'VLA tokens · smallest span {float(v):g} tokens'))
        self.pre_spin.committed.connect(lambda v: self._set_setting(
            'pre_grasp_m', float(v) / 1000.0, f'VLA tokens · pre-grasp distance {float(v):g} mm'))
        self.b_plucker = button('Export Plücker rays…', role='ghost', icon_name='export',
                                tip='The chosen policy\'s per-pixel rays (o × d, d) in the robot base frame at its '
                                    'input resolution: .npy, .safetensors and a provenance JSON per camera')
        self.b_plucker.clicked.connect(lambda: self.export_plucker())
        self.show3d = QCheckBox('Show token footprints in 3D')
        self.show3d.setChecked(True)
        self.show3d.toggled.connect(lambda _on: self._redraw())
        layout.addWidget(row(self.b_plucker, self.show3d))
        st = self.ctx.store
        st.changed.connect(lambda _s: self._project_changed())
        st.pose_changed.connect(lambda _uid: self._pose_changed())
        st.project_replaced.connect(self._replaced)
        self._sync_settings()

    # ---------------------------------------------------------------- settings
    def settings(self) -> dict:
        try:
            return VT.settings_of(self.ctx.store.project)
        except Exception:                               # noqa: BLE001 — a bad stored value: show the defaults
            return dict(VT.DEFAULTS, defaults=list(VT.DEFAULTS), policies=[], critical_objects=None, grasp_set='',
                        q={}, suggest=True)

    def _sync_settings(self):
        from ...core import grasps as GR
        st = self.settings()
        p = self.ctx.store.project
        self.side_spin.set_value(float(st['min_tokens_side']))
        self.side_def.setText('default' if 'min_tokens_side' in st['defaults'] else
                              f'yours (default {VT.DEFAULTS["min_tokens_side"]:g})')
        self.pre_spin.set_value(float(st['pre_grasp_m']) * 1000.0)
        self.pre_def.setText('default' if 'pre_grasp_m' in st['defaults'] else
                             f'yours (default {VT.DEFAULTS["pre_grasp_m"] * 1000:g} mm)')
        self.gs_box.blockSignals(True)
        self.gs_box.clear()
        self.gs_box.addItem('The first grasp set (default)', '')
        for gs in GR.sets(p):
            self.gs_box.addItem(f'{gs.get("name", "Grasp set")} · {len(gs.get("poses") or [])} grasps', gs['uid'])
        k = self.gs_box.findData(st['grasp_set'])
        self.gs_box.setCurrentIndex(k if k >= 0 else 0)
        self.gs_box.blockSignals(False)
        while self.crit_layout.count():
            w = self.crit_layout.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        crit = st['critical_objects']
        for i, o in enumerate(p.workspace.objects):
            cb = QCheckBox(o.name)
            cb.setChecked(crit is None or o.uid in crit)
            cb.setToolTip('A critical object warns when it spans too few tokens or falls outside the crop')
            cb.toggled.connect(lambda on, uid=o.uid: self.set_critical(uid, on))
            self.crit_layout.addWidget(cb, i // 3, i % 3)
        if not p.workspace.objects:
            self.crit_layout.addWidget(label('No task objects (Workspace ▸ Add task object).', muted=True), 0, 0)

    def _set_setting(self, key, value, text):
        st = self.settings()
        cur = st.get(key)
        if cur == value or (key == 'grasp_set' and (cur or None) == value):
            return

        def fn(p):
            d = dict(p.analysis.studies.get('vla_tokens') or {})
            if value is None:
                d.pop(key, None)
            else:
                d[key] = value
            p.analysis.studies['vla_tokens'] = d
        self.ctx.store.edit('analysis', fn, text, merge_key=('vla_tokens', key))

    def set_critical(self, uid: str, on: bool):
        """Mark a task object critical or not (one undo step); the default is every object."""
        p = self.ctx.store.project
        st = self.settings()
        cur = st['critical_objects']
        allu = [o.uid for o in p.workspace.objects]
        sel = list(allu if cur is None else cur)
        if on and uid not in sel:
            sel.append(uid)
        if not on and uid in sel:
            sel.remove(uid)
        new = None if set(sel) >= set(allu) else [u for u in allu if u in sel]
        if new == cur:
            return
        name = next((o.name for o in p.workspace.objects if o.uid == uid), 'object')
        self._set_setting('critical_objects', new, f'VLA tokens · {name} {"critical" if on else "not critical"}')

    # ---------------------------------------------------------------- run
    def live_q(self) -> dict:
        return {uid: [float(x) for x in np.asarray(q, float).reshape(-1)] for uid, q in self.ctx.store.q.items()}

    def run(self):
        """Run for the pose on screen (used when there is no grasp set; passed in the run's settings)."""
        self.ctx.analysis.run(self.STUDY, {'q': self.live_q()})

    def _project_changed(self):
        self._sync_settings()
        self._ov_key = None
        self._redraw()

    def _pose_changed(self):
        self._ov_key = None
        self._redraw()

    def _replaced(self):
        self.doc = None
        self._overlay3d, self._ov_key = None, None
        for w in (self.objs, self.tips):
            w.clear()
        for w in (self.mapping, self.warn_label, self.sugg_label, self.notes_label):
            w.setText('')
        self.pol_box.clear()
        self.cam_box.clear()
        self._sync_settings()
        self.suggestion_ready(False)
        self.empty()

    # ---------------------------------------------------------------- result
    def _headline(self, doc):
        s = doc.get('summary') or {}
        sev, word = s.get('status', 'good'), s.get('word', '')
        h = s.get('fewest')
        side = (doc.get('settings') or {}).get('min_tokens_side', VT.DEFAULTS['min_tokens_side'])
        tip = (f'A critical object warns below {side:g} × {side:g} tokens (' +
               ('the default' if 'min_tokens_side' in (doc.get('settings') or {}).get('defaults', []) else
                'your setting') + f') and when more than {VT.DEFAULTS["crop_warn_fraction"] * 100:g} % of it is '
               'cropped away (a default).')
        if h is None:
            ft = s.get('fingertips_fewest')
            if ft is not None:
                n = ft['tokens']
                self.head.set(str(n), 'token on a fingertip' if n == 1 else 'tokens on a fingertip', sev, word,
                              f'No task object is in a policy input, so the fingertips are counted: the fewest is '
                              f'{ft["fingertip"]} in {ft["camera"]} for {ft["policy"]} ({PHASE.get(ft["phase"])}). '
                              'Add the task objects (Workspace ▸ Add box object) to count what the policy must see.',
                              tip)
            else:
                self.head.set('—', '', sev, word, s.get('advice', ''), tip)
            return
        ext = h.get('extent') or [None, None]
        self.head.set(str(h['tokens']), 'token' if h['tokens'] == 1 else 'tokens', sev, word,
                      f'{h["object"]} in {h["camera"]} for {h["policy"]}: spans {_f(ext[0])} × {_f(ext[1])} tokens. '
                      + s.get('advice', ''), tip)

    def show_result(self, doc):
        self._headline(doc)
        cur_p, cur_c = self.pol_box.currentData(), self.cam_box.currentData()
        self.pol_box.blockSignals(True)
        self.pol_box.clear()
        for pr in doc.get('policies', []):
            self.pol_box.addItem(f'{pr["name"]} · {pr["input_size"][0]} × {pr["input_size"][1]}', pr['uid'])
        k = self.pol_box.findData(cur_p)
        self.pol_box.setCurrentIndex(k if k >= 0 else 0)
        self.pol_box.blockSignals(False)
        self._fill_cams(cur_c)
        warns = doc.get('warnings') or []
        self.warn_label.setText('\n'.join(f'{w["severity"].capitalize()}: {w["text"]}' for w in warns[:12]) +
                                (f'\n… and {len(warns) - 12} more (Export for all).' if len(warns) > 12 else '')
                                if warns else 'None.')
        sg = doc.get('suggestion')
        self.sugg_label.setText(('Suggestion: ' + sg['text']) if sg else '')
        self.suggestion_ready(bool(sg))
        self.notes_label.setText('\n'.join(str(n) for n in doc.get('notes') or []))
        self._ov_key = None
        self._redraw()

    def _policy(self):
        uid = self.pol_box.currentData()
        return next((p for p in (self.doc or {}).get('policies', []) if p['uid'] == uid), None)

    def _camera(self):
        pr = self._policy()
        uid = self.cam_box.currentData()
        return None if pr is None else next((c for c in pr['cameras'] if c['uid'] == uid), None)

    def _fill_cams(self, want=None):
        pr = self._policy()
        self.cam_box.blockSignals(True)
        self.cam_box.clear()
        for c in (pr or {}).get('cameras', []):
            self.cam_box.addItem(c['name'], c['uid'])
        k = self.cam_box.findData(want)
        self.cam_box.setCurrentIndex(k if k >= 0 else 0)
        self.cam_box.blockSignals(False)
        self._fill_tables()

    def _selection_changed(self, policy: bool):
        if policy:
            self._fill_cams(self.cam_box.currentData())
        else:
            self._fill_tables()
        self._ov_key = None
        self._redraw()

    def _fill_tables(self):
        pr, cr = self._policy(), self._camera()
        self.objs.clear()
        self.tips.clear()
        if pr is None:
            self.mapping.setText('')
            return
        g = pr.get('grid')
        if g:
            gtxt = (f'{g["kind"] if g["kind"] == "patch" else "token cell"} {g["cell_px"]} px → {g["cols"]} × '
                    f'{g["rows"]} = {g["cols"] * g["rows"]}')
            if g['kind'] != 'patch':
                gtxt += ' (derived from tokens per image)'
            if g.get('merged_cell_px'):
                gtxt += f'; merged tokens {g["merged_cell_px"]} px'
        else:
            gtxt = 'no token grid (neither patch nor tokens stated): pixels only'
        if cr is None:
            self.mapping.setText(f'{pr["name"]}: input {pr["input_size"][0]} × {pr["input_size"][1]}, '
                                 f'{pr["crop_label"]}, {gtxt}. No camera of this policy could be counted.')
            return
        m = cr['mapping']
        b = m['crop_box_sensor_px']
        txt = (f'{cr["name"]} ({cr["imager"]}, {cr["stream"][0]} × {cr["stream"][1]}, {cr["lens"]}) → {pr["name"]} '
               f'{pr["input_size"][0]} × {pr["input_size"][1]}, {pr["crop_label"]}\n'
               f'crop box u {b["u0"]:.1f} … {b["u1"]:.1f}, v {b["v0"]:.1f} … {b["v1"]:.1f} px '
               f'({b["width"]:.0f} × {b["height"]:.0f}); scale {m["scale_input_per_sensor_px"][0]:.4f} × '
               f'{m["scale_input_per_sensor_px"][1]:.4f} input px per sensor px; padding '
               f'{m["padded_fraction"] * 100:.0f} %\n{gtxt}')
        if m.get('note'):
            txt += f'\n{m["note"]}'
        self.mapping.setText(txt)
        src = pr.get('fields') or {}
        self.mapping.setToolTip('\n'.join(f'{k}: {v["source"]}' for k, v in src.items()) + '\n\n' + m['equation'])
        right = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        warned = {(w['object'], w['kind']) for w in pr.get('warnings', []) if w['sensor'] == cr['uid']}
        for o in cr['objects']:
            ext = o.get('worst_extent')
            st = 'not in view' if not o['poses_in_view'] else ('outside the crop' if not o['poses_in_input'] else
                                                                'ok')
            if (o['uid'], 'small') in warned:
                st = 'too few tokens'
            elif (o['uid'], 'cropped') in warned:
                st = 'partly cropped'
            if not o['critical']:
                st += ' (not critical)'
            it = QTreeWidgetItem([o['name'], _f(o.get('tokens_min'), '{:d}'), _f(o.get('tokens_half_min'), '{:d}'),
                                  '—' if not ext else f'{ext[0]:.1f} × {ext[1]:.1f}', _pct(o.get('outside_crop_max')),
                                  f'{o["poses_in_input"]} of {o["poses"]}', st])
            it.setToolTip(1, 'Tokens with at least one pixel of the object, at the pose with the fewest')
            it.setToolTip(2, 'Tokens at least half covered by the object, at the pose with the fewest')
            it.setToolTip(3, 'The object\'s bounding box in the input, in tokens (width × height), at its worst pose')
            it.setToolTip(4, 'The share of what the camera sees of the object that the crop cuts away (worst pose)')
            for k in range(1, 6):
                it.setFont(k, mono(11))
                it.setTextAlignment(k, right)
            if st.startswith(('too few', 'outside')):
                it.setForeground(6, QColor(STATUS['serious']))
            elif st.startswith('partly'):
                it.setForeground(6, QColor(STATUS['warning']))
            self.objs.addTopLevelItem(it)
        if not cr['objects']:
            self.objs.addTopLevelItem(QTreeWidgetItem(['No task objects', '', '', '', '', '', '']))
        for t in cr['fingertips']:
            it = QTreeWidgetItem([t['fingertip'], PHASE.get(t['phase'], t['phase']), _f(t['tokens_min'], '{:d}'),
                                  _f(t['tokens_median'], '{:g}'), _f(t['tokens_max'], '{:d}'),
                                  _pct(t['points_in_input_share'])])
            for k in range(2, 6):
                it.setFont(k, mono(11))
                it.setTextAlignment(k, right)
            self.tips.addTopLevelItem(it)
        _fit(self.objs)
        _fit(self.tips)

    # ---------------------------------------------------------------- apply / preview
    def apply(self):
        """Set the suggested stream on its camera (one undo step), then run again."""
        sg = (self.doc or {}).get('suggestion')
        if not sg:
            return False
        p = self.ctx.store.project
        s = p.variant.sensor(sg['sensor_uid'])
        if s is None:
            if self.ctx.window is not None:
                self.ctx.window.msg('That camera is no longer in the rig: run the study again.', 6000)
            return False
        mode = [int(sg['mode'][0]), int(sg['mode'][1])]

        def fn(proj):
            x = proj.variant.sensor(sg['sensor_uid'])
            if x is not None:
                x.stream = list(mode)
        self.ctx.store.edit('variant', fn, f'Set {s.name}\'s stream to {mode[0]} × {mode[1]} (VLA tokens)')
        self.suggestion_ready(False)
        self.run()
        return True

    def preview(self):
        sg = (self.doc or {}).get('suggestion')
        if sg and self.ctx.window is not None:
            self.ctx.window.msg('Preview: ' + sg['text'] + ' Apply makes the change (one undo step).', 12000)
        return sg

    # ---------------------------------------------------------------- export
    def export_plucker(self, folder: str | None = None):
        """Write the chosen policy's Plücker ray maps at the pose on screen into folder; the paths, or None."""
        from ...core import plucker as PL
        pr = self._policy()
        p = self.ctx.store.project
        uid = pr['uid'] if pr else (p.policies[0].uid if getattr(p, 'policies', None) else None)
        if uid is None:
            if self.ctx.window is not None:
                self.ctx.window.msg('There is no policy to export rays for: add one on the Policies page.', 6000)
            return None
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export Plücker rays to')
            if not folder:
                return None
        try:
            paths = PL.export_plucker(p, folder, uid, q=self.live_q() or None)
        except PL.PluckerError as e:
            if self.ctx.window is not None:
                self.ctx.window.msg(str(e), 8000)
            return None
        except Exception as e:                          # noqa: BLE001 — explained, logged
            traceback.print_exc()
            if self.ctx.window is not None:
                self.ctx.window.msg(f'The ray maps could not be written: {e}', 8000)
            return None
        if self.ctx.window is not None:
            n = sum(x.endswith('.json') for x in paths)
            self.ctx.window.msg(f'Wrote the Plücker rays of {n} camera{"s" if n != 1 else ""} ({len(paths)} files).',
                                6000)
        return paths

    # ---------------------------------------------------------------- overlays
    def _critical(self):
        p = self.ctx.store.project
        crit = self.settings()['critical_objects']
        return [o for o in p.workspace.objects if crit is None or o.uid in crit]

    def _live(self):
        """{sensor uid: (View, grid, [(object, tris, cells)])} at the pose on screen, cached until something moves."""
        from ...core import scenegeom as SG
        pr = self._policy()
        p = self.ctx.store.project
        uid = pr['uid'] if pr else None
        if uid is None:
            return {}
        key = (uid, self.ctx.store.revision, json.dumps(self.live_q(), sort_keys=True))
        if self._ov_key == key and self._live_cache is not None:
            return self._live_cache
        out = {}
        st = self.settings()
        try:
            objs = [(o, SG.object_triangles(o, p)) for o in self._critical()]
            for view, grid in VT.live_views(p, uid, {k: np.asarray(v) for k, v in self.ctx.store.q.items()}):
                cells = []
                for o, tris in objs:
                    if not len(tris):
                        continue
                    sil = VT.silhouette(view, tris, st['raster_tol_px'], st['max_subdiv'])
                    c = VT.count_mask(sil.input_mask, grid['cell_px'] if grid else None, sil.bbox)
                    cells.append((o, tris, c['cells'][:MAX_OVERLAY_CELLS]))
                out[view.sensor.uid] = (view, grid, cells)
        except Exception:                               # noqa: BLE001 — an overlay must not take the page down
            traceback.print_exc()
            out = {}
        self._ov_key, self._live_cache = key, out
        self._overlay3d = None
        return out

    _live_cache = None

    def overlays(self, vp) -> list:
        if not self.show3d.isChecked() or self.doc is None:
            return []
        from ..viewport3d import LineDraw
        live = self._live()
        if self._overlay3d is not None:
            return self._overlay3d
        items = []
        cr = self._camera()
        p = self.ctx.store.project
        for suid, (view, grid, cells) in live.items():
            if cr is not None and suid != cr['uid']:
                continue
            if grid is None:
                continue
            s = p.variant.sensor(suid)
            col = THEME.cam(s.slot) if s is not None else QColor(THEME.color('ink'))
            for _o, tris, cl in cells:
                try:
                    seg = VT.footprints(view, tris, grid['cell_px'], cl)
                except Exception:                       # noqa: BLE001
                    traceback.print_exc()
                    continue
                if len(seg):
                    items.append(LineDraw(seg.astype(np.float32), col, 1.2, segments=True, depth=False))
        self._overlay3d = items
        return items

    def camview_overlays(self, cam_model, pose) -> list:
        pr = self._policy()
        if pr is None:
            return []
        shown = ''
        try:
            shown = self.ctx.window.camview.view.sensor_uid or ''
        except AttributeError:
            pass
        live = self._live()
        if shown not in live:
            return []
        view, grid, cells = live[shown]
        if (cam_model.width, cam_model.height) != tuple(view.geom.src_size):
            # the Camera view shows another imager or size: map the crop onto it
            from ...core import policies as PO
            try:
                geom = PO.input_geometry(self.ctx.store.project, pr['uid'], (cam_model.width, cam_model.height))
            except PO.PolicyError:
                return []
        else:
            geom = view.geom
        s = self.ctx.store.project.variant.sensor(shown)
        col = THEME.cam(s.slot) if s is not None else QColor(THEME.color('ink'))
        w, h = geom.input_size
        out = [(VT.crop_polyline(geom), col, f'{pr["name"]} crop · input {w} × {h}')]
        if grid is not None:
            out.append((VT.grid_polylines(geom, grid['cell_px']), THEME.color('ink2'),
                        f'{"patch" if grid["kind"] == "patch" else "token"} grid {grid["cell_px"]} px'))
            if geom is view.geom:
                polys = []
                for _o, _t, cl in cells:
                    polys += VT.cell_polylines(geom, grid['cell_px'], cl)
                if polys:
                    out.append((polys, THEME.color('ink'), 'tokens on critical objects'))
        return out

    def _redraw(self):
        win = self.ctx.window
        if win is not None and hasattr(win, 'viewport'):
            win.viewport.update()
        cv = getattr(win, 'camview', None) if win is not None else None
        if cv is not None and hasattr(cv, 'view'):
            cv.view.refresh_overlays()

    # ---------------------------------------------------------------- explain
    def explain_text(self) -> str:
        st = self.settings()
        return (
            'A vision-language-action policy turns each camera image into its input (the crop rule and input '
            'resolution of its model card, or your measured figures on the Policies page) and slices it into square '
            'ViT patches. Each task object is projected through the camera\'s lens and the crop, rasterised at the '
            'input resolution, and the patches it touches are counted. Fingertips are counted from points on their '
            'contact pads. With a grasp set the counts are at its pre-grasp and contact poses (cameras on the arm '
            'are posed by IK); without one, at the pose of the run.\n\n'
            f'A critical object warns when it spans fewer than {st["min_tokens_side"]:g} × '
            f'{st["min_tokens_side"]:g} tokens (default 2 × 2, a DexVantage default, not a standard) or when the crop '
            f'cuts more than {st["crop_warn_fraction"] * 100:g} % of it away (default 1 %).\n\n'
            'Known limits: no occlusion test (the grasp self-occlusion study treats visibility); the inference-time '
            'crop only; tokens on a square grid; the fingertip pads at the home opening.')

    def explain(self):
        box = QMessageBox(self)
        box.setWindowTitle('How VLA token resolution is computed')
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


register('vla', 'VLA token resolution', VlaPage, menu='Analysis', icon='checker', workspace='grasp', order=34)
