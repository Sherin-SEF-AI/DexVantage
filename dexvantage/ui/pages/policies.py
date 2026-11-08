"""The Policies page (spec §13, §5, §1.1.6): the policies the data is for, each a policy library entry (ACT, Diffusion
Policy, π0, π0-FAST, SmolVLA, OpenVLA, Octo, GR00T N1: figures from the model card, the paper or the official code,
each cited with the date read), the cameras it takes, and the user's own figures over the entry's.

Every field shows its value and where it comes from; a figure no source states is "—" ("not stated by the source").
Overriding a field records the user's figure with its source ("measured by the user: ..."); clearing the override
brings back the entry's figure. Adding, renaming, removing, assigning cameras and each override are one undo step each
(the project's 'policies' section).

Camera view: while this page is shown, the selected policy's input region is drawn on the cameras it takes: the part
of the image the policy sees after its crop rule (core/policies.py crop_geometry), labelled with the input size.

Known limits: the page shows the inference-time crop, not training augmentation; per-camera-role inputs (Octo's wrist
image) are listed, not drawn per camera; the figures are those of the configuration each source publishes (enter a
fine-tune's own as overrides).
"""
from __future__ import annotations

import traceback

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QHeaderView, QLineEdit,
                             QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ...core import policies as PO
from ..theme import THEME, mono
from ..widgets import Section, button, label
from . import register
from .base import StudyPage, row

ROLE = Qt.ItemDataRole.UserRole


class PoliciesPage(StudyPage):
    """Policies: the robot-learning policies the data is for, each with its input resolution, crop rule, ViT patch,
    action chunk, temporal-ensemble weight and inference time from its cited source, or your own measured figure.
    The headline counts the figures the studies need that no source states: measure them and enter them here."""
    STUDY = None
    TITLE = 'Policies'
    MODEL = ('Model: the policy figures as resolved from their model card or paper, or your measurement '
             '(core/policies.py).')
    HEADLINE = 'Policy figures not stated'

    def __init__(self, ctx):
        self._uid = ''
        self._building = False
        super().__init__(ctx)
        for b in (self.b_run, self.b_apply, self.b_preview):
            b.hide()
        st = ctx.store
        st.changed.connect(self._changed)
        st.project_replaced.connect(self.refresh)
        self.refresh()

    # ---------------------------------------------------------------- layout
    def build_body(self, layout):
        self.add_box = QComboBox()
        self.add_box.setToolTip('A policy from the library: its figures come from its model card, paper or code')
        self.b_add = button('Add policy', icon_name='plus', tip='Add the chosen library policy (one undo step)')
        self.b_add.clicked.connect(lambda: self.add_policy(self.add_box.currentData()))
        self.b_del = button('Remove', role='ghost', icon_name='trash', tip='Remove the selected policy')
        self.b_del.clicked.connect(lambda: self.remove())
        layout.addWidget(row(self.add_box, self.b_add, self.b_del))
        self.table = QTreeWidget()
        self.table.setColumnCount(5)
        self.table.setHeaderLabels(['Policy', 'Library entry', 'Cameras', 'Input', 'Not stated'])
        self.table.setRootIsDecorated(False)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        hdr = self.table.header()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for k in (2, 3, 4):
            hdr.setSectionResizeMode(k, QHeaderView.ResizeMode.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._picked)
        layout.addWidget(self.table)
        self.editor_host = QWidget()
        self.editor_lay = QVBoxLayout(self.editor_host)
        self.editor_lay.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.editor_host)

    def empty(self):
        """No study: refresh fills the headline."""

    # ---------------------------------------------------------------- state
    def current(self):
        ps = self.ctx.store.project.policies
        p = PO.policy(self.ctx.store.project, self._uid) or (ps[0] if ps else None)
        self._uid = p.uid if p is not None else ''
        return p

    def _changed(self, sections):
        if sections & {'policies', 'variant', 'variants'}:
            self.refresh()

    def _picked(self):
        if self._building:
            return
        its = self.table.selectedItems()
        if its and its[0].data(0, ROLE) and its[0].data(0, ROLE) != self._uid:
            self._uid = its[0].data(0, ROLE)
            self.refresh()

    # ---------------------------------------------------------------- filling
    def refresh(self):
        self._building = True
        try:
            p = self.ctx.store.project
            self.add_box.clear()
            for e in PO.library_entries():
                self.add_box.addItem(f'{e.name} · {e.maker}' + (' (custom)' if e.is_custom else ''), e.id)
                self.add_box.setItemData(self.add_box.count() - 1, e.note, Qt.ItemDataRole.ToolTipRole)
            cur = self.current()
            self.table.clear()
            for pol in p.policies:
                det = PO.resolve_detail(p, pol.uid)
                cams = PO.policy_cameras(p, pol)
                missing = [k for k in PO.NEEDED if det[k]['value'] is None]
                cam_text = (', '.join(s.name for s in cams) or 'none') if pol.cameras else f'all ({len(cams)})'
                it = QTreeWidgetItem([pol.name, PO.title_of(pol), cam_text,
                                      PO.format_value('input_resolution', det['input_resolution']['value']),
                                      str(len(missing))])
                it.setData(0, ROLE, pol.uid)
                for k in (3, 4):
                    it.setFont(k, mono(11))
                if missing:
                    it.setToolTip(4, 'Not stated: ' + ', '.join(PO.in_sentence(PO.FIELDS[k].label) for k in missing))
                self.table.addTopLevelItem(it)
                if cur is not None and pol.uid == cur.uid:
                    it.setSelected(True)
            rows = max(1, min(self.table.topLevelItemCount(), 8))
            self.table.setFixedHeight(self.table.header().sizeHint().height() + rows * 22 + 6)
            self.b_del.setEnabled(cur is not None)
            self.doc = PO.summary(p)
            self._headline()
            self._build_editor(cur)
        except Exception:                                 # noqa: BLE001 — explained, the traceback logged
            traceback.print_exc()
            self.head.set('—', '', 'critical', 'could not show', 'The policies could not be shown: details are in '
                          'the log.')
        finally:
            self._building = False
        if self.ctx.window is not None and getattr(self.ctx.window, 'camview', None) is not None:
            try:
                self.ctx.window.camview.view.schedule()
            except Exception:                             # noqa: BLE001 — a redraw request must not fail the page
                pass

    def _headline(self):
        p = self.ctx.store.project
        if not p.policies:
            self.head.set('—', '', 'warning', 'no policy', 'Add the policy the data is for: its input size, crop '
                          'rule and timing drive the token, latency and chunk studies.')
            return
        gaps = [(pol, k) for pol in p.policies for k, d in PO.resolve_detail(p, pol.uid).items()
                if k in PO.NEEDED and d['value'] is None]
        cams = self._camera_mismatch()
        self.head.title.setText(self.HEADLINE)
        if gaps:
            pol, k = gaps[0]
            more = f' and {len(gaps) - 1} more' if len(gaps) > 1 else ''
            self.head.set(str(len(gaps)), 'not stated', 'warning', 'measure them',
                          f'{pol.name}: {PO.in_sentence(PO.FIELDS[k].label)} is not stated by its source{more}. '
                          'Measure it on your rig and enter it below as an override, with how you measured it.')
        elif cams:
            text, k, n = cams                             # every figure is cited: the camera count is what matters
            self.head.title.setText('Policy cameras assigned')
            self.head.set(f'{k} of {n}', 'cameras assigned', 'warning', 'too few' if k < n else 'too many', text,
                          'Every figure the studies need is cited or measured; the camera count does not match.')
        else:
            self.head.set('0', 'not stated', 'good', 'all cited',
                          'Every figure the studies need is cited from its source or measured by you.')

    def _camera_mismatch(self):
        """(sentence, cameras assigned, images the policy takes) for the first policy whose counts differ, else None."""
        p = self.ctx.store.project
        for pol in p.policies:
            n = PO.resolve(p, pol.uid)['n_cameras'][0]
            k = len(PO.policy_cameras(p, pol))
            if n is not None and k != n:
                return (f'{pol.name} takes {n} camera image{"s" if n != 1 else ""} per step; {k} '
                        f'camera{"s are" if k != 1 else " is"} assigned. '
                        + ('Add a camera to the rig and assign it below.' if k < n else 'Choose its cameras below.'),
                        k, n)
        return None

    def _build_editor(self, pol):
        while self.editor_lay.count():
            w = self.editor_lay.takeAt(0).widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        if pol is None:
            return
        try:
            self.editor_lay.addWidget(self._editor(pol))
        except Exception:                                 # noqa: BLE001 — explained, the traceback logged
            traceback.print_exc()
            self.editor_lay.addWidget(label(f'{pol.name} cannot be edited here: details are in the log.', wrap=True))

    def _editor(self, pol) -> QWidget:
        p = self.ctx.store.project
        sec = Section(pol.name)
        self.name_edit = QLineEdit(pol.name)
        self.name_edit.editingFinished.connect(lambda: self.rename(pol.uid, self.name_edit.text()))
        sec.row('Name', self.name_edit)
        # cameras
        self.cam_all = QCheckBox('Every camera and depth sensor of the rig')
        self.cam_all.setChecked(not pol.cameras)
        self.cam_all.toggled.connect(lambda on: self.set_cameras(pol.uid, [] if on else
                                                                 [s.uid for s in PO.policy_cameras(p, pol)]))
        sec.row('Cameras', self.cam_all)
        self.cam_boxes = {}
        for s in p.variant.sensors:
            if s.kind not in ('camera', 'depth'):
                continue
            b = QCheckBox(s.name)
            b.setChecked(s.uid in pol.cameras or not pol.cameras)
            b.setEnabled(bool(pol.cameras))
            b.toggled.connect(lambda _on, uid=pol.uid: self.set_cameras(
                uid, [u for u, x in self.cam_boxes.items() if x.isChecked()]))
            self.cam_boxes[s.uid] = b
            sec.row('', b)
        # the fields
        self.fields = QTreeWidget()
        self.fields.setColumnCount(3)
        self.fields.setHeaderLabels(['Field', 'Value', 'Source'])
        self.fields.setRootIsDecorated(False)
        hdr = self.fields.header()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        det = PO.resolve_detail(p, pol.uid)
        for k, f in PO.FIELDS.items():
            d = det[k]
            val = PO.format_value(k, d['value'])
            if d.get('hardware') and k == 'inference_ms':
                val += f' ({d["hardware"]})'
            src = {'library': 'cited', 'user': 'yours', 'not_stated': PO.NOT_STATED}[d['origin']]
            it = QTreeWidgetItem([f.label, val, src + ('' if d['origin'] == 'not_stated' else ': ' + d['source'])])
            it.setFont(1, mono(11))
            it.setData(0, ROLE, k)
            it.setToolTip(0, f.text)
            tip = d['source'] + (f'\n\nThe library\'s figure: {PO.format_value(k, d.get("library_value"))} — '
                                 f'{d.get("library_source")}' if d['origin'] == 'user' else '')
            it.setToolTip(1, tip)
            it.setToolTip(2, tip)
            if d['origin'] == 'not_stated':
                it.setForeground(1, THEME.color('muted'))
                it.setForeground(2, THEME.color('muted'))
            self.fields.addTopLevelItem(it)
        self.fields.setFixedHeight(self.fields.header().sizeHint().height() + len(PO.FIELDS) * 22 + 6)
        self.fields.itemSelectionChanged.connect(self._field_picked)
        sec.add(self.fields)
        # the override
        self.ov_field = QComboBox()
        for k, f in PO.FIELDS.items():
            self.ov_field.addItem(f.label, k)
        self.ov_value = QLineEdit()
        self.ov_value.setPlaceholderText('value, e.g. 95 or 224 × 224')
        self.ov_source = QLineEdit()
        self.ov_source.setPlaceholderText('how you measured it, on what hardware')
        self.b_set = button('Set', role='primary', tip='Use your figure for this field (one undo step)')
        self.b_set.clicked.connect(lambda: self.set_override(pol.uid, self.ov_field.currentData(),
                                                             self.ov_value.text(), self.ov_source.text()))
        self.b_clear = button('Clear', role='ghost', tip='Use the library\'s figure again (one undo step)')
        self.b_clear.clicked.connect(lambda: self.clear_override(pol.uid, self.ov_field.currentData()))
        sec.row('Override', self.ov_field, tip='Your own figure, measured, over the library\'s')
        sec.row('Value', self.ov_value)
        sec.row('Source', self.ov_source, tip='Recorded as "measured by the user: ..."')
        sec.row('', self.b_set, self.b_clear, stretch_last=False)
        self.ov_message = label('', muted=True, wrap=True, small=True)
        sec.add(self.ov_message)
        e = PO._entry(pol.library_id)
        if e is not None:
            t = label(e.note, muted=True, wrap=True, small=True)
            t.setToolTip('\n'.join(f'[{i}] {s.get("title", "")}, {s.get("url", "")}, read {s.get("read", "")}'
                                   for i, s in enumerate(e.sources)))
            sec.add(t)
        return sec

    def _field_picked(self):
        its = self.fields.selectedItems()
        if its:
            i = self.ov_field.findData(its[0].data(0, ROLE))
            if i >= 0:
                self.ov_field.setCurrentIndex(i)

    # ---------------------------------------------------------------- edits (one undo step each)
    def _say(self, text, ms=6000):
        if getattr(self, 'ov_message', None) is not None:
            try:
                self.ov_message.setText(text)
            except RuntimeError:                          # the editor was rebuilt
                pass
        if self.ctx.window is not None:
            self.ctx.window.msg(text, ms)

    def _edit(self, uid, fn, text) -> bool:
        st = self.ctx.store
        pol = PO.policy(st.project, uid)
        if pol is None:
            return False

        def wrap(p):
            fn(PO.policy(p, uid))
        st.edit('policies', wrap, text.replace('{name}', pol.name))
        return True

    def add_policy(self, library_id: str | None = None):
        """A policy of a library entry (one undo step); returns it."""
        from ...core.project import Policy
        e = PO._entry(library_id or '')
        if e is None:
            self._say('Choose a policy from the library first.')
            return None
        p = self.ctx.store.project
        names = {x.name for x in p.policies}
        name, i = e.name, 2
        while name in names:
            name, i = f'{e.name} {i}', i + 1
        pol = Policy(name=name, library_id=e.id)
        self.ctx.store.edit('policies', lambda pr: pr.policies.append(pol), f'Added the policy {name}')
        self._uid = pol.uid
        self.refresh()
        return PO.policy(self.ctx.store.project, pol.uid)

    def remove(self, uid: str | None = None) -> bool:
        uid = uid or self._uid
        pol = PO.policy(self.ctx.store.project, uid)
        if pol is None:
            return False
        self.ctx.store.edit('policies', lambda pr: setattr(pr, 'policies', [x for x in pr.policies if x.uid != uid]),
                            f'Removed the policy {pol.name}')
        self._uid = ''
        self.refresh()
        return True

    def rename(self, uid, text):
        t = ' '.join((text or '').split())
        pol = PO.policy(self.ctx.store.project, uid)
        if t and pol is not None and t != pol.name:
            self._edit(uid, lambda o: setattr(o, 'name', t), f'Renamed the policy {{name}} to {t}')

    def set_cameras(self, uid, sensor_uids) -> bool:
        """The cameras the policy takes ([] = every camera and depth sensor)."""
        pol = PO.policy(self.ctx.store.project, uid)
        if self._building or pol is None:
            return False
        valid = {s.uid for s in self.ctx.store.project.variant.sensors}
        cams = [u for u in dict.fromkeys(sensor_uids or []) if u in valid]
        if cams == pol.cameras:
            return False
        text = 'every camera' if not cams else f'{len(cams)} camera{"s" if len(cams) != 1 else ""}'
        ok = self._edit(uid, lambda o: setattr(o, 'cameras', cams), f'Set {{name}} to take {text}')
        self.refresh()
        return ok

    def set_override(self, uid, field: str, value, source: str) -> bool:
        """Use the user's figure for a field, with where it comes from (one undo step); False with the reason said
        when the figure or its source cannot be used."""
        pol = PO.policy(self.ctx.store.project, uid)
        if pol is None:
            return False
        try:
            v = PO.check_value(field, value if not isinstance(value, str) or value.strip() else None)
            if v is None:
                raise PO.PolicyError('Enter a value: to use the library\'s figure again, clear the override.')
            src = PO.user_source(source)
        except PO.PolicyError as e:
            self._say(str(e))
            return False
        label_ = PO.in_sentence(PO.FIELDS[field].label)

        def fn(o):
            o.overrides = dict(o.overrides, **{field: {'value': v, 'source': src}})
        self._edit(uid, fn, f'Set {{name}}\'s {label_} to {PO.format_value(field, v)} (measured by you)')
        self.refresh()
        return True

    def clear_override(self, uid, field: str) -> bool:
        pol = PO.policy(self.ctx.store.project, uid)
        if pol is None or field not in pol.overrides:
            self._say('That field has no override of yours to clear.')
            return False
        label_ = PO.in_sentence(PO.FIELDS.get(field, PO.Field(field, '', '', '')).label)
        self._edit(uid, lambda o: setattr(o, 'overrides', {k: v for k, v in o.overrides.items() if k != field}),
                   f'Cleared {{name}}\'s {label_} override')
        self.refresh()
        return True

    # ---------------------------------------------------------------- export, camera view
    def export(self, path=None):
        if path is None:
            if not self.ctx.store.project.policies:
                self._say('There is no policy to export yet: add one first.', 5000)
                return None
            path, _ = QFileDialog.getSaveFileName(self, 'Export policies', 'policies.json', 'JSON (*.json)')
            if not path:
                return None
        self.doc = PO.summary(self.ctx.store.project)
        return super().export(path)

    def camview_overlays(self, cam_model, pose) -> list:
        p = self.ctx.store.project
        pol = PO.policy(p, self._uid)
        if pol is None:
            return []
        shown = ''
        try:
            shown = self.ctx.window.camview.view.sensor_uid or ''
        except AttributeError:
            pass
        if shown and shown not in {s.uid for s in PO.policy_cameras(p, pol)}:
            return []
        W, H = int(cam_model.width), int(cam_model.height)
        try:
            g = PO.input_geometry(p, pol.uid, (W, H))
        except PO.PolicyError:
            return []
        x0, y0, x1, y1 = (v - 0.5 for v in g.src_rect)        # pixel edges -> OpenCV pixel-centre coordinates
        rect = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], float)
        w, h = g.input_size
        txt = f'{pol.name} input {w} × {h} · {PO.CROP_LABEL.get(g.rule, g.rule)}'
        if g.padded_fraction > 1e-6:
            txt += f' · {g.padded_fraction * 100:.0f} % padding'
        s = p.variant.sensor(shown) if shown else None
        return [(rect, THEME.cam(s.slot) if s is not None else THEME.color('ink'), txt)]


register('policies', 'Policies', PoliciesPage, menu='Tasks', icon='target', order=25)
