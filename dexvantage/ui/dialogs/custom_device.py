"""Library ▸ New custom device…: a form for a device of your own (camera, depth camera, tactile sensor or host), every
figure marked as given (by the maker or a measurement of yours) or not given, saved to your library
($XDG_DATA_HOME/DexVantage/library, core/library.add_custom) and validated against the library schema.

Ported from Vantage ui/dialogs/hardware_dialog.py (a spec sheet with units, required fields, live validation that
blocks saving, "start from" a library item). What changed: the fields are those of DexVantage's library schema
(resources/library.schema.json); a figure that is not ticked as given is stored as null and shown everywhere as "—"
with "not stated by the maker", never filled in; a source (URL or document, the date read) is required, as for the
built-in entries.

Starting from a library device copies all of its figures and sources (its other imagers, stream tables and notes
included) into a new custom device: the form edits the first imager and the device-wide figures.
"""
from __future__ import annotations

import copy
import datetime as _dt
import math
import re
from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLineEdit, QPlainTextEdit, QScrollArea,
                             QVBoxLayout, QWidget)

from ...core import library
from ..rigedit import device_title
from ..widgets import button, label
from .base import VDialog

KIND_LABEL = {'camera': 'Camera', 'depth': 'Depth camera', 'tactile': 'Tactile sensor', 'host': 'Host'}
CHOICES = {
    'im_role': ('rgb', 'mono', 'ir_left', 'ir_right', 'tof'),
    'shutter': ('global', 'rolling'),
    'lens_model': library.LENS_MODELS,
    'focus': ('fixed', 'auto'),
    'technology': ('active_ir_stereo', 'passive_stereo', 'tof', 'structured_light'),
    'host_type': ('usb_controller', 'pcie_card', 'gmsl_deser', 'compute'),
}
_ALL = ('camera', 'depth', 'tactile', 'host')
_IMG = ('camera', 'depth')


@dataclass
class Fig:
    key: str
    text: str
    unit: str = ''
    type: str = 'float'        # float | int | text | choice | pair | ipair | triple | floats
    kinds: tuple = _ALL
    required: bool = False
    help: str = ''


FIELDS = (
    # (section title, [Fig])
    ('Device', [
        Fig('maker', 'Maker', type='text', required=True),
        Fig('name', 'Name', type='text', required=True),
        Fig('id', 'Library id', type='text', required=True,
            help='Lower-case letters, digits, ".", "_" and "-"; filled from the maker and name until you change it'),
        Fig('family', 'Family', type='text'),
        Fig('interface', 'Interface', type='text', help='As the maker names it, e.g. "USB 3.1 Gen 1" or "GMSL2"'),
        Fig('power_w', 'Power', 'W'),
        Fig('mass_g', 'Mass', 'g'),
        Fig('size_mm', 'Size', 'mm (x, y, z)', 'triple'),
    ]),
    ('Source', [
        Fig('src_url', 'URL or document', type='text', required=True,
            help='Where the figures come from: the datasheet\'s URL, or the document\'s name and number, or '
                 '"measured by me" with how'),
        Fig('src_title', 'Title', type='text'),
        Fig('src_read', 'Date read', 'YYYY-MM-DD', 'text', required=True),
        Fig('src_covers', 'What it gives', type='text', help='Which of the figures this source states'),
    ]),
    ('Imager', [
        Fig('im_name', 'Imager name', type='text', kinds=_IMG, required=True),
        Fig('im_role', 'Role', type='choice', kinds=_IMG, required=True),
        Fig('shutter', 'Shutter', type='choice', kinds=_IMG),
        Fig('hfov', 'Horizontal FOV', '°', kinds=_IMG, help='Edge to edge of the native image, as stated'),
        Fig('vfov', 'Vertical FOV', '°', kinds=_IMG),
        Fig('dfov', 'Diagonal FOV', '°', kinds=_IMG),
        Fig('resolution', 'Resolution', 'px (w, h)', 'ipair', kinds=_IMG, help='The native image the FOV is for'),
        Fig('pixel_um', 'Pixel size', 'µm', kinds=_IMG),
        Fig('lens_model', 'Lens model', type='choice', kinds=_IMG,
            help='pinhole unless the maker states a distortion model; a calibration gives the real one'),
        Fig('f_number', 'f-number', kinds=_IMG),
        Fig('focus', 'Focus', type='choice', kinds=_IMG),
        Fig('fps', 'Frame rates', 'fps (list)', 'floats', kinds=_IMG,
            help='At the resolution above; a stream mode is made from them'),
    ]),
    ('Depth', [
        Fig('technology', 'Technology', type='choice', kinds=('depth',)),
        Fig('baseline_mm', 'Stereo baseline', 'mm', kinds=('depth',),
            help='Given: a second imager ("IR right") is placed one baseline to the right of the first'),
        Fig('max_range_m', 'Maximum range', 'm', kinds=('depth',)),
        Fig('min_z_mm', 'Minimum Z at the resolution above', 'mm', kinds=('depth',)),
        Fig('rms_text', 'Depth error, as stated', type='text', kinds=('depth',),
            help='The maker\'s own words, e.g. "<2% at 2 m"'),
    ]),
    ('Tactile', [
        Fig('pad_mm', 'Pad size', 'mm (w, h)', 'pair', kinds=('tactile',)),
        Fig('tcam_res', 'Internal camera', 'px (w, h)', 'ipair', kinds=('tactile',)),
        Fig('tfps', 'Frame rate', 'fps', kinds=('tactile',)),
        Fig('sensing', 'Sensing', type='text', kinds=('tactile',), help='e.g. optical, piezoresistive'),
        Fig('taxels', 'Taxels', type='int', kinds=('tactile',)),
    ]),
    ('Host', [
        Fig('host_type', 'Type', type='choice', kinds=('host',)),
        Fig('ports', 'Ports', type='int', kinds=('host',)),
        Fig('lanes', 'Lanes', type='int', kinds=('host',)),
        Fig('stated_mbps', 'Stated bandwidth', 'Mb/s', kinds=('host',)),
        Fig('practical_mbps', 'Practical bandwidth', 'Mb/s', kinds=('host',),
            help='Only when a source states it (a measurement of yours counts: say so in the source)'),
        Fig('gmsl_links', 'GMSL links', type='int', kinds=('host',)),
    ]),
)
FIG = {f.key: f for _t, fs in FIELDS for f in fs}
_OPTICAL_Q = [-0.5, 0.5, -0.5, 0.5]        # the optical frame of an imager looking along body x (REP-103)


def slug(text: str) -> str:
    s = re.sub(r'[^a-z0-9]+', '-', (text or '').lower()).strip('-')
    return s or 'device'


def parse(f: Fig, text: str):
    """The value of a field's text; raises ValueError with a plain message."""
    t = (text or '').strip()
    if f.type in ('text', 'choice'):
        if not t:
            raise ValueError(f'{f.text}: is empty')
        return t
    if not t:
        raise ValueError(f'{f.text}: is ticked as given but has no value')
    parts = [p for p in re.split(r'[\s,;×x]+', t.replace('−', '-')) if p]
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f'{f.text}: "{t}" is not a number' + ('' if f.type in ('float', 'int') else 's')) from None
    if not all(math.isfinite(v) for v in nums):          # 'nan' and 'inf' parse as floats
        raise ValueError(f'{f.text}: "{t}" is not a finite number')
    want = {'float': 1, 'int': 1, 'pair': 2, 'ipair': 2, 'triple': 3}.get(f.type)
    if want is not None and len(nums) != want:
        raise ValueError(f'{f.text}: needs {want} number{"s" if want > 1 else ""}, got {len(nums)}')
    if f.type in ('int', 'ipair') and any(v != int(v) for v in nums):
        raise ValueError(f'{f.text}: needs whole numbers')
    if any(v < 0 for v in nums) or (f.type in ('float', 'pair', 'ipair', 'triple', 'floats')
                                     and any(v <= 0 for v in nums)):
        # a figure of 0 (a 0 mm baseline, a 0 m range, a 0° FOV) is not a figure: untick it when it is not given
        raise ValueError(f'{f.text}: must be more than 0' if f.type != 'int' else f'{f.text}: must be positive')
    if f.type == 'float':
        return nums[0]
    if f.type == 'int':
        return int(nums[0])
    if f.type == 'ipair':
        return [int(v) for v in nums]
    return nums


def fmt(v) -> str:
    if v is None:
        return ''
    if isinstance(v, (list, tuple)):
        return ', '.join(fmt(x) for x in v)
    if isinstance(v, float):
        return f'{v:g}'
    return str(v)


class CustomDeviceDialog(VDialog):
    def __init__(self, ctx, parent=None, kind: str = 'camera', start_from: str | None = None):
        super().__init__(parent, 'New custom device', 'Every figure marked as given or not given', (720, 760))
        self.ctx = ctx
        self.base: dict | None = None
        self.rows: dict = {}              # key -> (Fig, given QCheckBox | None, editor, row widget, label)
        self._id_edited = False
        self.saved = None
        lay = QVBoxLayout(self.body)
        lay.setContentsMargins(14, 10, 14, 10)
        top = QHBoxLayout()
        top.addWidget(label('Kind', ink2=True))
        self.kind_box = QComboBox()
        for k in _ALL:
            self.kind_box.addItem(KIND_LABEL[k], k)
        self.kind_box.currentIndexChanged.connect(lambda _i: self._kind_changed())
        top.addWidget(self.kind_box, 1)
        self.from_box = QComboBox()
        self.from_box.setToolTip('Start from a library device: its figures and sources are copied into the form')
        self.from_box.activated.connect(lambda i: self.start_from(self.from_box.itemData(i)) if i > 0 else None)
        top.addWidget(self.from_box, 2)
        lay.addLayout(top)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        lay.addWidget(self.scroll, 1)
        self.msg = label('', wrap=True)
        self.msg.setTextFormat(Qt.TextFormat.PlainText)
        lay.addWidget(self.msg)
        self._build_form()
        self.b_save = button('Save to my library', 'primary')
        self.b_save.clicked.connect(self.save)
        b_close = button('Close', 'ghost')
        b_close.clicked.connect(self.reject)
        self.buttons.addStretch(1)
        self.buttons.addWidget(b_close)
        self.buttons.addWidget(self.b_save)
        self.hint.setText('Untick a figure the maker does not state: it is stored as not given, never guessed.')
        self.kind_box.setCurrentIndex(max(0, self.kind_box.findData(kind)))
        self._kind_changed()
        if start_from:
            self.start_from(start_from)

    # ------------------------------------------------------------------ the form
    def _build_form(self):
        w = QWidget()
        form = QFormLayout(w)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.sections = {}
        for title, figs in FIELDS:
            head = label(title, bold=True)
            form.addRow(head)
            self.sections[title] = (head, figs)
            for f in figs:
                row = QWidget()
                h = QHBoxLayout(row)
                h.setContentsMargins(0, 0, 0, 0)
                h.setSpacing(6)
                given = None
                if not f.required:
                    given = QCheckBox('given')
                    given.setToolTip('Ticked: the source states this figure. Unticked: not given (stored as null, '
                                     f'shown as "—" with "{library.NOT_STATED}")')
                    given.toggled.connect(lambda on, k=f.key: self._given(k, on))
                    h.addWidget(given)
                if f.type == 'choice':
                    ed = QComboBox()
                    for c in CHOICES[f.key]:
                        ed.addItem(c, c)
                    ed.currentIndexChanged.connect(self._check)
                else:
                    ed = QLineEdit()
                    ed.setPlaceholderText('not given' if not f.required else 'required')
                    ed.textChanged.connect(self._check)
                ed.setToolTip(f.help or f.text)
                h.addWidget(ed, 1)
                if f.unit:
                    u = label(f.unit, muted=True, small=True)
                    h.addWidget(u)
                lab = label(f.text + (' *' if f.required else ''), ink2=not f.required)
                lab.setToolTip(f.help or f.text)
                form.addRow(lab, row)
                self.rows[f.key] = (f, given, ed, row, lab)
                if given is not None:
                    ed.setEnabled(False)
        self.note = QPlainTextEdit()
        self.note.setPlaceholderText('Anything the figures need: assumptions (labelled as such), what the source does '
                                     'not state, units it uses…')
        self.note.setFixedHeight(80)
        form.addRow(label('Note', ink2=True), self.note)
        self.scroll.setWidget(w)
        self.editor('src_read').setText(_dt.date.today().isoformat())
        self.editor('im_name').setText('RGB')
        self.editor('maker').textChanged.connect(self._auto_id)
        self.editor('name').textChanged.connect(self._auto_id)
        self.editor('id').textEdited.connect(lambda _t: setattr(self, '_id_edited', True))

    def editor(self, key):
        return self.rows[key][2]

    def _given(self, key, on):
        self.rows[key][2].setEnabled(on)
        self._check()

    def set_given(self, key, on: bool):
        g = self.rows[key][1]
        if g is not None:
            g.setChecked(bool(on))

    def set_value(self, key, value):
        """Fill a field (None = not given)."""
        f, g, ed, _r, _l = self.rows[key]
        if isinstance(ed, QComboBox):
            if value is not None:
                ed.setCurrentIndex(max(0, ed.findData(value)))
        else:
            ed.setText(fmt(value))
        if g is not None:
            g.setChecked(value is not None)

    def kind(self) -> str:
        return self.kind_box.currentData()

    def _kind_changed(self):
        k = self.kind()
        for title, (head, figs) in self.sections.items():
            vis = any(k in f.kinds for f in figs)
            head.setVisible(vis)
            for f in figs:
                _f, _g, _ed, row, lab = self.rows[f.key]
                row.setVisible(k in f.kinds)
                lab.setVisible(k in f.kinds)
        if k == 'depth' and self.editor('im_name').text() in ('', 'RGB'):
            self.editor('im_name').setText('IR left')
            self.set_value('im_role', 'ir_left')
        elif k == 'camera' and self.editor('im_name').text() in ('', 'IR left'):
            self.editor('im_name').setText('RGB')
            self.set_value('im_role', 'rgb')
        self.from_box.blockSignals(True)
        self.from_box.clear()
        self.from_box.addItem('Start from a library device…', None)
        for e in library.entries(k):
            self.from_box.addItem(device_title(e) + (' (custom)' if e.is_custom else ''), e.id)
        self.from_box.blockSignals(False)
        self._check()

    def _auto_id(self, *_):
        if not self._id_edited:
            t = slug(f'{self.editor("maker").text()} {self.editor("name").text()}')
            self.editor('id').setText(t if library.get(t) is None or library.get(t).is_custom else t + '-custom')

    # ------------------------------------------------------------------ start from a library device
    def start_from(self, model_id):
        e = library.get(model_id)
        if e is None:
            return
        d = copy.deepcopy(e.data)
        self.base = d
        if self.kind() != e.kind:
            self.kind_box.setCurrentIndex(self.kind_box.findData(e.kind))
        self.set_value('maker', d.get('maker'))
        self.set_value('name', f'{d.get("name")} (custom)')
        self._id_edited = True
        self.editor('id').setText(f'{e.id}-custom' if not e.is_custom else e.id)
        for k in ('family', 'interface', 'power_w', 'mass_g', 'size_mm'):
            self.set_value(k, d.get(k))
        self.editor('src_url').setText('')
        self.editor('src_title').setText('')
        self.set_given('src_title', False)
        self.set_given('src_covers', False)
        ims = d.get('imagers') or []
        if ims:
            im = ims[0]
            self.set_value('im_name', im.get('name'))
            self.set_value('im_role', im.get('role'))
            for k, src in (('shutter', 'shutter'), ('hfov', 'hfov_deg'), ('vfov', 'vfov_deg'), ('dfov', 'dfov_deg'),
                           ('resolution', 'resolution'), ('pixel_um', 'pixel_um'), ('lens_model', 'lens_model'),
                           ('f_number', 'f_number'), ('focus', 'focus')):
                self.set_value(k, im.get(src))
            fps = sorted({f for s in e.streams(im.get('name')) for f in s.get('fps') or []})
            self.set_value('fps', fps or None)
        dp = d.get('depth') or {}
        self.set_value('technology', dp.get('technology'))
        b = dp.get('baseline_m')
        self.set_value('baseline_mm', None if b is None else round(b * 1000, 6))
        self.set_value('max_range_m', dp.get('max_range_m'))
        res = ims[0].get('resolution') if ims else None
        mz = next((r.get('m') for r in dp.get('min_z') or [] if res and [r['width'], r['height']] == list(res)), None)
        self.set_value('min_z_mm', None if mz is None else round(mz * 1000, 6))
        self.set_value('rms_text', (dp.get('rms_error') or {}).get('text'))
        t = d.get('tactile') or {}
        for k, src in (('pad_mm', 'pad_mm'), ('tcam_res', 'camera_resolution'), ('tfps', 'fps'),
                       ('sensing', 'sensing'), ('taxels', 'taxels')):
            self.set_value(k, t.get(src))
        h = d.get('host') or {}
        for k, src in (('host_type', 'type'), ('ports', 'ports'), ('lanes', 'lanes'), ('stated_mbps', 'stated_mbps'),
                       ('practical_mbps', 'practical_mbps'), ('gmsl_links', 'gmsl_links')):
            self.set_value(k, h.get(src))
        self.note.setPlainText(f'Started from the library entry {e.id}; its sources are kept below this one.')
        self._check()

    # ------------------------------------------------------------------ values
    def values(self) -> tuple[dict, list]:
        """({key: value or None (not given)}, [plain error messages])."""
        out, errs = {}, []
        k = self.kind()
        for key, (f, g, ed, _r, _l) in self.rows.items():
            if k not in f.kinds:
                continue
            if g is not None and not g.isChecked():
                out[key] = None
                continue
            text = ed.currentData() if isinstance(ed, QComboBox) else ed.text()
            try:
                out[key] = parse(f, text)
            except ValueError as e:
                out[key] = None
                errs.append(str(e))
        return out, errs

    def entry(self) -> tuple[dict, list]:
        """(the library entry the form describes, [plain error messages])."""
        v, errs = self.values()
        k = self.kind()
        base = copy.deepcopy(self.base) if self.base and self.base.get('kind') == k else {}
        not_given = [FIG[key].text for key, val in v.items() if val is None and not FIG[key].required
                     and not key.startswith('src_')]
        src = {'url': v.get('src_url') or '', 'read': v.get('src_read') or ''}
        if v.get('src_title'):
            src['title'] = v['src_title']
        if v.get('src_covers'):
            src['covers'] = v['src_covers']
        if src['read'] and not re.match(r'^\d{4}-\d{2}-\d{2}$', src['read']):
            errs.append('Date read: write it as YYYY-MM-DD')
        note = self.note.toPlainText().strip()
        auto = (f'Custom device. Not given: {", ".join(not_given)}.' if not_given else 'Custom device.')
        d = dict(base)
        d.update({'id': v.get('id') or '', 'kind': k, 'maker': v.get('maker') or '', 'name': v.get('name') or '',
                  'sources': [src] + [s for s in base.get('sources') or [] if s.get('url') != src['url']],
                  'note': (note + ' ' if note else '') + auto,
                  'interface': v.get('interface'), 'power_w': v.get('power_w'), 'mass_g': v.get('mass_g'),
                  'size_mm': v.get('size_mm')})
        if v.get('family') is not None:
            d['family'] = v['family']
        else:
            d.pop('family', None)
        if k in _IMG:
            ims = copy.deepcopy(base.get('imagers') or [])
            im = ims[0] if ims else {'pose': {'position': [0.0, 0.0, 0.0], 'quaternion': list(_OPTICAL_Q)},
                                     'pose_stated': True}
            old = im.get('name')
            im.update({'name': v.get('im_name') or '', 'role': v.get('im_role'), 'shutter': v.get('shutter'),
                       'hfov_deg': v.get('hfov'), 'vfov_deg': v.get('vfov'), 'dfov_deg': v.get('dfov'),
                       'resolution': v.get('resolution'), 'pixel_um': v.get('pixel_um'),
                       'lens_model': v.get('lens_model'), 'f_number': v.get('f_number'), 'focus': v.get('focus')})
            ims = [im] + ims[1:]
            streams = copy.deepcopy(base.get('streams') or [])
            if old and old != im['name']:
                for s in streams:
                    if s.get('imager') == old:
                        s['imager'] = im['name']
            res = v.get('resolution')
            if res and v.get('fps'):
                streams = [s for s in streams if not (s.get('imager') in (im['name'], 'depth')
                                                      and [s['width'], s['height']] == list(res))]
                streams.insert(0, {'imager': im['name'], 'width': res[0], 'height': res[1], 'fps': v['fps'],
                                   'format': None, 'bits_per_pixel': None})
                if k == 'depth':
                    streams.insert(1, {'imager': 'depth', 'width': res[0], 'height': res[1], 'fps': v['fps'],
                                       'format': None, 'bits_per_pixel': None})
            d['imagers'] = ims
            d['streams'] = streams
        else:
            d['imagers'] = []
            d['streams'] = copy.deepcopy(base.get('streams') or [])
        if k == 'depth':
            dp = copy.deepcopy(base.get('depth') or {})
            b = v.get('baseline_mm')
            ims = d['imagers']
            if b is not None and len(ims) == 1:
                right = copy.deepcopy(ims[0])
                right.update({'name': 'IR right', 'role': 'ir_right',
                              'pose': {'position': [0.0, -b / 1000, 0.0], 'quaternion': list(_OPTICAL_Q)},
                              'pose_stated': True})
                ims.append(right)
            pair = [ims[0]['name'], ims[1]['name']] if len(ims) > 1 and b is not None else None
            mz = list(dp.get('min_z') or [])
            res = v.get('resolution')
            if res is not None:
                mz = [r for r in mz if [r['width'], r['height']] != list(res)]
                if v.get('min_z_mm') is not None:
                    mz.insert(0, {'width': res[0], 'height': res[1], 'm': v['min_z_mm'] / 1000})
            elif v.get('min_z_mm') is not None:
                errs.append('Minimum Z: give the resolution it is for (Imager ▸ Resolution)')
            rms = None
            if v.get('rms_text'):
                rms = dict(dp.get('rms_error') or {}, text=v['rms_text'])
                rms.setdefault('pct', None)
                rms.setdefault('at_m', None)
            dp.update({'technology': v.get('technology'), 'reference_imager': ims[0]['name'] if ims else None,
                       'stereo_pair': pair if pair else dp.get('stereo_pair') if len(ims) > 1 else None,
                       'baseline_m': None if b is None else b / 1000, 'min_z': mz,
                       'max_range_m': v.get('max_range_m'), 'rms_error': rms})
            dp.setdefault('subpixel_px', None)
            dp.setdefault('depth_fov', None)
            d['depth'] = dp
        else:
            d['depth'] = None
        if k != 'depth':
            d['projector'] = None
        else:
            d.setdefault('projector', None)
        d['tactile'] = ({'pad_mm': v.get('pad_mm'), 'camera_resolution': v.get('tcam_res'), 'fps': v.get('tfps'),
                         'sensing': v.get('sensing'), 'taxels': v.get('taxels')} if k == 'tactile' else None)
        d['host'] = ({'type': v.get('host_type'), 'ports': v.get('ports'), 'lanes': v.get('lanes'),
                      'stated_mbps': v.get('stated_mbps'), 'practical_mbps': v.get('practical_mbps'),
                      'gmsl_links': v.get('gmsl_links')} if k == 'host' else None)
        if k in _IMG and not errs:
            errs += _lens_errors(v)
        old = library.get(d['id']) if d['id'] else None
        if old is not None and not old.is_custom:
            errs.append(f'Library id: "{d["id"]}" is a built-in device; give yours an id of its own')
        if not errs:
            errs += [_plain(m) for m in library.validate(d)]
        return d, errs

    def _check(self, *_):
        if not hasattr(self, 'b_save'):                  # still building the form
            return []
        _d, errs = self.entry()
        self.b_save.setEnabled(not errs)
        self.msg.setText('\n'.join(f'✗ {e}' for e in errs[:8]) + (f'\n… and {len(errs) - 8} more'
                                                                    if len(errs) > 8 else '')
                         if errs else 'Ready to save to your library.')
        return errs

    # ------------------------------------------------------------------ save
    def save(self):
        """Validate and write the device to the user library; the Library panel shows it at once. Returns the Entry
        (None when the form is not valid, with the reasons shown)."""
        d, errs = self.entry()
        if errs:
            self._check()
            return None
        try:
            e = library.add_custom(d)
        except library.LibraryError as ex:
            self.msg.setText(str(ex))
            return None
        self.saved = e
        win = self.ctx.window
        if win is not None and hasattr(win, 'library'):
            win.library.refill()
        if win is not None and hasattr(win, 'msg'):
            win.msg(f'Saved {device_title(e)} to your library', 5000)
        self._kind_changed()                              # refills "start from" (and re-checks the form) first,
        self.msg.setText(f'Saved {device_title(e)} to your library ({e.id}). Double-click it in the Library to add '
                         'it to the rig.')                # so this confirmation is what stays on screen
        return e

    @classmethod
    def run(cls, ctx, parent=None, kind: str = 'camera', start_from: str | None = None):
        """Show the dialog (modeless, so the rig stays usable while it is filled)."""
        d = cls(ctx, parent, kind=kind, start_from=start_from)
        d.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        d.show()
        return d


def _lens_errors(v: dict) -> list:
    """The imager's stated FOV against its lens model, as the library will build it (core/optics from_fov): a
    pinhole cannot show 180° or more, and the FOVs must fit one image. [] when it can be modelled."""
    from ...core.optics import CameraModel
    hf, vf, df = v.get('hfov'), v.get('vfov'), v.get('dfov')
    model = v.get('lens_model') or 'pinhole'
    errs = []
    if model == 'pinhole':
        for key, val in (('hfov', hf), ('vfov', vf), ('dfov', df)):
            if val is not None and val >= 180:
                errs.append(f'{FIG[key].text}: a pinhole lens shows less than 180°; choose a fisheye lens model '
                            '(equidistant, equisolid, stereographic or Kannala-Brandt) for a wider lens')
    if errs or hf is None:
        return errs
    w, h = v.get('resolution') or library.DEFAULT_RESOLUTION
    try:
        CameraModel.from_fov(int(w), int(h), float(hf), None if vf is None else float(vf), model=model)
    except ValueError as e:
        errs.append(f'Field of view: {e}')
    return errs


_SCHEMA_PATHS = tuple((re.compile(rx), text) for rx, text in (
    (r'^imagers\[\d+\]\.hfov_deg', FIG['hfov'].text), (r'^imagers\[\d+\]\.vfov_deg', FIG['vfov'].text),
    (r'^imagers\[\d+\]\.dfov_deg', FIG['dfov'].text), (r'^imagers\[\d+\]\.resolution', FIG['resolution'].text),
    (r'^imagers\[\d+\]\.pixel_um', FIG['pixel_um'].text), (r'^imagers\[\d+\]\.f_number', FIG['f_number'].text),
    (r'^imagers\[\d+\]\.name', FIG['im_name'].text), (r'^streams\[\d+\]\.fps', FIG['fps'].text),
    (r'^depth\.baseline_m', FIG['baseline_mm'].text), (r'^depth\.max_range_m', FIG['max_range_m'].text),
    (r'^depth\.min_z', 'Minimum Z'), (r'^id\b', FIG['id'].text), (r'^sources\[0\]\.read', FIG['src_read'].text),
    (r'^sources\[0\]\.url', FIG['src_url'].text)))


def _plain(msg: str) -> str:
    """A schema message in plainer words: the form's field name for the schema path, and the bound in words."""
    where, sep, what = msg.partition(': ')
    if sep:
        for rx, text in _SCHEMA_PATHS:
            if rx.match(where):
                where = text
                break
    what = re.sub(r'^\S+ is less than or equal to the minimum of (\S+)$', r'must be more than \1', what)
    what = re.sub(r'^\S+ is less than the minimum of (\S+)$', r'must be at least \1', what)
    what = re.sub(r'^\S+ is greater than or equal to the maximum of (\S+)$', r'must be less than \1', what)
    what = re.sub(r'^\S+ is greater than the maximum of (\S+)$', r'must be at most \1', what)
    what = re.sub(r"^'.*' does not match '.*'$", 'has characters that are not allowed', what)
    what = what.replace("None is not of type 'string'", 'is required').replace(
        'is not valid under any of the given schemas', 'is not a valid value')
    return f'{where}: {what}' if sep else what


def open_dialog(window, kind: str = 'camera', start_from: str | None = None):
    """Library ▸ New custom device… from the main window."""
    return CustomDeviceDialog.run(window.ctx, window, kind=kind, start_from=start_from)
