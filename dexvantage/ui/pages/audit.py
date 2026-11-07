"""The Dataset audit page (spec §8.2): a recorded dataset's episodes and campaign checked against the requirements
(core/studies/audit.py, core/audit).

What it shows: the episodes that meet every requirement (the headline, with a status word: ready when all do, mostly
usable from 80 %, needs work from 50 %, not usable below; DexVantage defaults), the dataset picker (the datasets of
project.data, Add folder… / Add file… for a bag), the requirement table (limit, labelled default or yours, checks
passed, the worst value, the episodes failing), the episode table (verdict, faults, each camera's offset), the
selected episode's cameras (frames, measured fps, drops, gaps, duplicates, jitter, offset ± σ, exposure steps, blur,
gripper hidden), the campaign chart of each camera's mount over the episodes (± σ, the steps found marked), and the
requirement settings. Apply records the episodes to drop or re-record in the dataset's entry (data.datasets[i].exclude,
one undo step); Preview says which; Export writes the result as JSON with provenance; Export report… writes JSON and
HTML; Export to FiftyOne… writes a FiftyOne dataset folder (core/audit/report.py). The 3D view draws each audited
camera's viewing direction in its status colour; the Camera view draws, for a camera whose mount moved, where the
planned image border sits after the move.

Known limits: the page shows one dataset at a time; the audit runs in the background (long); see the study's limits.
"""
from __future__ import annotations

import os
import threading
import traceback

import numpy as np
from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (QComboBox, QFileDialog, QGridLayout, QHeaderView, QMessageBox, QTreeWidget,
                             QTreeWidgetItem, QWidget)

from ...core.audit import episode as AE
from ...core.datasets.episode import format_title
from ...core.studies import audit as AU
from ..theme import STATUS
from ..widgets import ScrubSpinBox, button, label
from . import register
from ._curves import CurveChart
from .base import StudyPage, row

VERDICT_SEV = {'pass': 'good', 'fix': 'warning', 'recalibrate': 'warning', 'drop': 'critical'}
EDITABLE = (('max_offset_ms', 'Camera-joint offset', 'ms', 1), ('max_jitter_ms', 'Timestamp jitter', 'ms', 2),
            ('max_dropped_frames', 'Dropped frames', 'frames', 0), ('max_mount_shift_deg', 'Mount slip', '°', 2),
            ('blur_ratio', 'Blur ratio', '', 2), ('gripper_ncc', 'Gripper correlation', '', 2))
PALETTE = ('#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300')


def _f(x, nd=1, unit='') -> str:
    if x is None:
        return '—'
    if isinstance(x, (list, tuple)):
        return '–'.join(_f(v, nd) for v in x) + (f' {unit}' if unit else '')
    if isinstance(x, int):
        if x == 1 and unit in ('frames', 'gaps', 'steps'):        # "1 gap", not "1 gaps"
            unit = unit[:-1]
        return f'{x:,}' + (f' {unit}' if unit else '')
    return f'{x:.{nd}f}' + (f' {unit}' if unit else '')


class _Sig(QObject):
    opened = pyqtSignal(object)
    fp = pyqtSignal(object)


class AuditPage(StudyPage):
    """Dataset audit: every episode of a recorded dataset checked from its own timestamps, pixels and joint states
    (dropped and duplicated frames, frame rate, USB gaps, jitter, camera–joint offset and drift, exposure steps, motion
    blur, the gripper hidden in a wrist camera, action alignment), and the campaign's mounts (slip against gradual
    drift). The figure is the number of episodes that meet every requirement; the requirements are defaults you can
    change."""
    STUDY = AU.SID
    TITLE = 'Dataset audit'
    HEADLINE = 'Episodes that meet every requirement'

    def build_body(self, layout):
        self._sig = _Sig()
        self._sig.opened.connect(self._opened)
        self._sig.fp.connect(self._run_with)
        self._episode = None
        self.ds_box = QComboBox()
        self.ds_box.setMinimumWidth(240)
        self.ds_box.setToolTip('The dataset this page audits (the project\'s datasets, kept with the project)')
        self.ds_box.setPlaceholderText('No dataset yet: Add folder…')
        self.ds_box.activated.connect(lambda _i: self._choose(self.ds_box.currentData()))
        self.b_add = button('Add folder…', role='ghost', icon_name='open',
                            tip='A LeRobot or RLDS dataset folder, or a ROS 2 bag folder')
        self.b_add_file = button('Add file…', role='ghost', tip='A ROS 2 bag file (.mcap or .db3)')
        self.b_add_hub = button('From the Hub…', role='ghost',
                                tip='A public LeRobot dataset on the Hugging Face Hub, by name (org/name, optionally '
                                    '@revision): its files are fetched as the audit reads them and kept in a local '
                                    'cache')
        self.b_add_hub.clicked.connect(lambda: self.add_hub_dataset())
        self.b_remove = button('Remove', role='ghost', tip='Take the dataset out of the project (one undo step)')
        self.b_add.clicked.connect(lambda: self.add_dataset())
        self.b_add_file.clicked.connect(lambda: self.add_dataset(file=True))
        self.b_remove.clicked.connect(self.remove_dataset)
        self.b_report = button('Export report…', role='ghost', icon_name='export', tip='JSON and HTML, with provenance')
        self.b_fo = button('Export to FiftyOne…', role='ghost', icon_name='export',
                           tip='A FiftyOne dataset folder: per-episode tags and fields, per-frame scores (FiftyOne '
                               'itself is not needed to write it)')
        self.b_report.clicked.connect(lambda: self.export_report())
        self.b_fo.clicked.connect(lambda: self.export_fiftyone())
        layout.addWidget(row(label('Dataset'), self.ds_box, self.b_add, self.b_add_file, self.b_add_hub,
                             self.b_remove))
        layout.addWidget(row(self.b_report, self.b_fo))
        self.ds_label = label('', muted=True, small=True, wrap=True)
        layout.addWidget(self.ds_label)
        layout.addWidget(label('Requirements', bold=True))
        self.reqs = QTreeWidget()
        self.reqs.setRootIsDecorated(False)
        self.reqs.setHeaderLabels(['Requirement', 'Limit', 'Checks passed', 'Worst', 'Episodes failing'])
        _columns(self.reqs, 4)
        layout.addWidget(self.reqs)
        layout.addWidget(label('Episodes (click one for its cameras)', bold=True))
        self.eps = QTreeWidget()
        self.eps.setRootIsDecorated(False)
        self.eps.setUniformRowHeights(True)
        self.eps.setMinimumHeight(160)
        self.eps.setHeaderLabels(['Episode', 'Length', 'Verdict', 'Faults', 'Camera offsets'])
        _columns(self.eps, 4)
        self.eps.itemClicked.connect(lambda it, _c: self.show_episode(it.data(0, Qt.ItemDataRole.UserRole)))
        layout.addWidget(self.eps)
        self.cams = QTreeWidget()
        self.cams.setRootIsDecorated(False)
        self.cams.setHeaderLabels(['Camera', 'Frames', 'fps', 'Dropped', 'Gaps', 'Duplicated', 'Jitter (ms)',
                                   'Offset (ms)', 'Exposure steps', 'Blurred', 'Gripper hidden'])
        _columns(self.cams, 0)
        layout.addWidget(self.cams)
        layout.addWidget(label('Mount over the campaign', bold=True))
        self.chart = CurveChart(height=200)
        layout.addWidget(self.chart)
        self.chart_note = label('', muted=True, small=True, wrap=True)
        layout.addWidget(self.chart_note)
        layout.addWidget(label('Requirements you can change (defaults labelled)', bold=True))
        g = QWidget()
        grid = QGridLayout(g)
        grid.setContentsMargins(0, 0, 0, 0)
        self.spins = {}
        for i, (k, title, unit, nd) in enumerate(EDITABLE):
            sp = ScrubSpinBox('', 'plain', nd, unit, step=1.0 if nd == 0 else 0.1 if nd == 1 else 0.05, width=100)
            sp.setRange(0.0, 100000.0)
            d = AE.REQUIREMENTS[k][0] if k in AE.REQUIREMENTS else AE.SETTINGS[k]
            sp.setToolTip(f'{title}: default {d} {unit} (a DexVantage default, not a standard)')
            sp.committed.connect(lambda v, k=k, t=title: self._set_requirement(k, v, t))
            self.spins[k] = sp
            grid.addWidget(label(title), i // 3, 2 * (i % 3))
            grid.addWidget(sp, i // 3, 2 * (i % 3) + 1)
        layout.addWidget(g)
        self.defaults_label = label('', muted=True, small=True, wrap=True)
        layout.addWidget(self.defaults_label)
        self.notes_label = label('', muted=True, wrap=True)
        layout.addWidget(self.notes_label)
        st = self.ctx.store
        st.changed.connect(lambda _s: self._sync())
        st.project_replaced.connect(self._replaced)
        self._sync()

    # ---------------------------------------------------------------- datasets
    def _datasets(self) -> list:
        return list((getattr(self.ctx.store.project, 'data', None) or {}).get('datasets') or [])

    def _current_uid(self):
        st = self.ctx.store.project.analysis.studies.get(AU.SID) or {}
        uid = st.get('dataset_uid')
        ds = self._datasets()
        if uid and any(d['uid'] == uid for d in ds):
            return uid
        return ds[0]['uid'] if ds else None

    def _sync(self):
        cur = self._current_uid()
        self.ds_box.blockSignals(True)
        self.ds_box.clear()
        for d in self._datasets():
            self.ds_box.addItem(f'{d.get("name") or os.path.basename(d["path"])} · '
                                f'{format_title(d.get("format", ""), d.get("version", ""))}', d['uid'])
        if cur is not None:
            self.ds_box.setCurrentIndex(max(0, self.ds_box.findData(cur)))
        self.ds_box.blockSignals(False)
        self.b_remove.setEnabled(cur is not None)
        e = next((d for d in self._datasets() if d['uid'] == cur), None)
        self.ds_label.setText('' if e is None else f'{e.get("episodes") or "?"} episodes'
                              + (f'; {len(e["exclude"])} marked to drop or re-record' if e.get('exclude') else ''))
        try:
            req = AE.requirements_of(self.ctx.store.project)
        except Exception:                               # noqa: BLE001 — a bad stored value: show the defaults
            req = AE.requirements_of(None)
        for k, sp in self.spins.items():
            sp.blockSignals(True)
            sp.setValue(float(req[k]))
            sp.blockSignals(False)
        d = [k for k in req['defaults'] if k in self.spins]
        self.defaults_label.setText(('Defaults in use: ' + ', '.join(d) + ' (DexVantage defaults, not standards).')
                                    if d else 'Every requirement shown is yours.')

    def _choose(self, uid):
        if uid is None or uid == self._current_uid():
            return

        def fn(p):
            d = dict(p.analysis.studies.get(AU.SID) or {})
            d['dataset_uid'] = uid
            d.pop('dataset', None)
            p.analysis.studies[AU.SID] = d
        self.ctx.store.edit('analysis', fn, 'Dataset audit · dataset chosen')

    def add_dataset(self, path: str | None = None, file: bool = False):
        """Open a dataset in the background (its format and episodes), then add it to the project (one undo step)."""
        if path is None:
            if file:
                path, _ = QFileDialog.getOpenFileName(self, 'Add a ROS 2 bag', '', 'ROS 2 bags (*.mcap *.db3)')
            else:
                path = QFileDialog.getExistingDirectory(self, 'Add a dataset folder')
            if not path:
                return None

        def job(progress, cancel):
            from ...core.datasets import open_dataset
            progress(0.2, 'Reading the dataset')
            with open_dataset(path) as ds:
                return {'path': path, 'format': ds.format, 'version': ds.version, 'episodes': len(ds.episodes())}
        win = self.ctx.window
        if win is not None and hasattr(win, 'run_job'):
            win.run_job('audit_open', 'Opening the dataset', job, self._sig.opened.emit,
                        lambda m: win.msg(f'The dataset could not be opened: {m.splitlines()[0]}', 10000))
            return path
        try:
            self._opened(job(lambda *_: None, lambda: False))
        except Exception as e:                          # noqa: BLE001
            traceback.print_exc()
            return str(e)
        return path

    def add_hub_dataset(self, name: str | None = None):
        """Add a LeRobot dataset on the Hugging Face Hub by name ('org/name', 'org/name@revision', an hf:// reference
        or its huggingface.co URL); it is opened in the background like a folder."""
        from ...core.datasets.hub import hub_ref, parse_hub_ref
        if name is None:
            from PyQt6.QtWidgets import QInputDialog
            name, ok = QInputDialog.getText(self, 'Add a dataset from the Hugging Face Hub',
                                            'Dataset name (org/name, optionally @revision):')
            if not ok or not name.strip():
                return None
        name = name.strip()
        ref = parse_hub_ref(name)
        if ref is None and '/' in name and '://' not in name:
            repo, _, rev = name.partition('@')
            ref = parse_hub_ref(hub_ref(repo, rev or 'main'))
        if ref is None:
            if self.ctx.window is not None:
                self.ctx.window.msg(f'"{name}" is not a Hub dataset name: write it as org/name (for example '
                                    'lerobot/svla_so100_pickplace), optionally followed by @revision.', 8000)
            return None
        return self.add_dataset(hub_ref(*ref))

    def _opened(self, info):
        from ...core.project import new_uid
        from ...core.datasets.hub import parse_hub_ref
        uid = new_uid()
        hub = parse_hub_ref(info['path'])
        name = (hub[0] + ('' if hub[1] == 'main' else f'@{hub[1]}')) if hub else \
            os.path.basename(os.path.normpath(info['path']))

        def fn(p):
            p.data = dict(p.data)
            p.data['datasets'] = list(p.data.get('datasets') or []) + [
                {'uid': uid, 'name': name, 'path': info['path'], 'format': info['format'], 'version': info['version'],
                 'episodes': info['episodes']}]
            d = dict(p.analysis.studies.get(AU.SID) or {})
            d['dataset_uid'] = uid
            d.pop('dataset', None)
            p.analysis.studies[AU.SID] = d
        self.ctx.store.edit(['data', 'analysis'], fn, f'Dataset audit · added {name}')
        if self.ctx.window is not None:
            n = info['episodes']
            self.ctx.window.msg(f'Added {name}: {format_title(info["format"], info["version"])}, {n} '
                                f'episode{"" if n == 1 else "s"}. Run the audit to check it.', 6000)

    def remove_dataset(self):
        uid = self._current_uid()
        if uid is None:
            return

        def fn(p):
            p.data = dict(p.data)
            p.data['datasets'] = [d for d in p.data.get('datasets') or [] if d['uid'] != uid]
        self.ctx.store.edit('data', fn, 'Dataset audit · dataset removed')

    def _set_requirement(self, key, value, title):
        v = int(round(value)) if isinstance(AE.REQUIREMENTS.get(key, (0.0,))[0], int) else float(value)

        def fn(p):
            p.data = dict(p.data)
            r = dict(p.data.get('requirements') or {})
            r[key] = v
            p.data['requirements'] = r
        self.ctx.store.edit('data', fn, f'Dataset audit · {title.lower()} {v:g}', merge_key=('audit_req', key))

    # ---------------------------------------------------------------- run (with the dataset's fingerprint)
    def run(self):
        try:
            path = AU.dataset_path(self.ctx.store.project)
        except Exception as e:                          # noqa: BLE001 — no dataset yet: say so in the headline
            self.head.set('—', '', 'warning', 'no dataset', str(e))
            return
        threading.Thread(target=lambda: self._sig.fp.emit(AU.fingerprint(path)), daemon=True).start()

    def _run_with(self, fp):
        self.ctx.analysis.run(AU.SID, {'fingerprint': fp})

    def _replaced(self):
        self.doc = None
        for t in (self.reqs, self.eps, self.cams):
            t.clear()
        self.chart.set_data([], (0, 1), (0, 1))
        self.suggestion_ready(False)
        self._sync()
        self.empty()

    # ---------------------------------------------------------------- result
    def show_result(self, doc):
        try:
            s = doc.get('summary') or {}
            sev, word = AU.status_for(doc)
            n = s.get('episodes', 0)
            ds = doc.get('dataset') or {}
            self.head.set(f'{s.get("pass", 0)} / {n}', 'episodes', sev, word, s.get('advice', ''),
                          f'{ds.get("name", "")}: {ds.get("format", "")} {ds.get("version", "")}, '
                          f'{ds.get("episodes_audited", 0)} of {ds.get("episodes_total", 0)} episodes audited. Status: '
                          'ready when every episode passes, mostly usable from 80 %, needs work from 50 % '
                          '(DexVantage defaults).')
            self._fill_requirements(doc)
            self._fill_episodes(doc)
            self._fill_chart(doc)
            eps = doc.get('episodes') or []
            if eps:
                self.show_episode(0)
            notes = list(doc.get('plan_notes') or [])
            for k, m in (doc.get('mounts') or {}).items():
                notes += [f'{k}: {x}' for x in m.get('notes') or []][:3]
            self.notes_label.setText('\n'.join(notes))
        except (KeyError, TypeError, ValueError):
            traceback.print_exc()
        self.suggestion_ready(self.suggestion() is not None)
        self._redraw()

    def _fill_requirements(self, doc):
        self.reqs.clear()
        for d in doc.get('requirements') or []:
            lim, worst, unit = d.get('limit'), d.get('worst'), d.get('unit') or ''
            limit = '—' if lim is None else f'{_f(lim, 2)} {unit}' + (' (default)' if d.get('default') else
                                                                      ' (yours)')
            # a figure past the limit that still passes is within 3 σ of zero (not resolved), so not a failure
            unresolved = d['pass'] and isinstance(lim, (int, float)) and isinstance(worst, (int, float)) and \
                abs(worst) > lim
            it = QTreeWidgetItem([d['title'], limit, f'{d["checked"] - d["failed"]} of {d["checked"]}',
                                  _f(worst, 2, unit) + (' (not resolved)' if unresolved else ''),
                                  ', '.join(str(x) for x in d['episodes_failed'][:12]) or '—'])
            it.setForeground(2, QColor(STATUS['good' if d['pass'] else 'critical']))
            if unresolved:
                it.setToolTip(3, 'The largest estimate is past the limit but within 3 σ of zero, so it is not '
                                 'counted as a failure: the episode is too short to resolve it.')
            self.reqs.addTopLevelItem(it)
        _fit(self.reqs)

    def _fill_episodes(self, doc):
        self.eps.clear()
        for i, e in enumerate(doc.get('episodes') or []):
            offs = '; '.join(f'{k.replace("observation.images.", "")} {_f(c["sync"].get("offset_ms"))} ± '
                             f'{_f(c["sync"].get("sigma_ms"))}' for k, c in e['cameras'].items())
            it = QTreeWidgetItem([str(e['index']), f'{e["duration_s"]:.2f} s', e['verdict'],
                                  ', '.join(e['flags']) or '—', offs])
            it.setForeground(2, QColor(STATUS[VERDICT_SEV.get(e['verdict'], 'warning')]))
            it.setToolTip(2, (doc.get('verdict_text') or {}).get(e['verdict'], ''))
            it.setData(0, Qt.ItemDataRole.UserRole, i)
            self.eps.addTopLevelItem(it)

    def show_episode(self, i):
        eps = (self.doc or {}).get('episodes') or []
        if i is None or not 0 <= int(i) < len(eps):
            return
        self._episode = int(i)
        e = eps[self._episode]
        self.cams.clear()
        for key, c in e['cameras'].items():
            tm, sy = c['timing'], c['sync']
            it = QTreeWidgetItem([key.replace('observation.images.', ''), _f(tm['frames']),
                                  _f(tm.get('fps_measured'), 2), _f(tm.get('dropped_frames', 0)),
                                  _f(len(tm['usb_gaps'])), _f(len(c['duplicates']['frames'])),
                                  _f(tm.get('jitter_ms'), 2),
                                  f'{_f(sy.get("offset_ms"))} ± {_f(sy.get("sigma_ms"))}',
                                  _f(len(c['exposure']['steps'])),
                                  'not tested' if c['blur'].get('skipped') else _f(len(c['blur']['frames'])),
                                  _f(len(c['gripper'].get('hidden') or [])) if c['gripper'].get('tested') else '—'])
            it.setToolTip(7, f'Against {sy.get("reference", "")}; ρ {_f(sy.get("rho"), 2)}. Drift '
                          f'{_f(sy.get("drift_ppm"), 0)} ± {_f(sy.get("drift_sigma_ppm"), 0)} ppm.')
            if c['blur'].get('skipped'):
                it.setToolTip(9, c['blur']['skipped'])
            self.cams.addTopLevelItem(it)
        _fit(self.cams)

    def _fill_chart(self, doc):
        series, xs, ys, vlines = [], [], [], []
        for k, (key, m) in enumerate((doc.get('mounts') or {}).items()):
            S = m.get('episodes') or []
            if len(S) < 2:
                continue
            R = np.array([[np.nan if v is None else v for v in s['rotation_deg']] for s in S], float)
            if not np.isfinite(R).any():
                continue
            ax = int(np.nanargmax(np.nanmax(np.abs(R), axis=0)))
            t = [float(s['t']) for s in S]
            y = [None if not np.isfinite(v) else float(v) for v in R[:, ax]]
            col = QColor(PALETTE[k % len(PALETTE)])
            name = f'{key.replace("observation.images.", "")} ({("roll", "pitch", "yaw")[ax]})'
            series.append((name, t, y, col, 'line'))
            series.append((name, t, y, col, 'points'))
            xs += t
            ys += [v for v in y if v is not None]
            for ev in m.get('events') or []:
                if ev['kind'] in ('step', 'excursion') and 'tau' in ev:
                    vlines.append((float(ev['tau']), f'{key.replace("observation.images.", "")} '
                                                     f'{ev["kind"]} {ev["size"]:+.2f}{ev["unit"]}'))
        if not series:
            self.chart.set_data([], (0, 1), (0, 1))
            self.chart_note.setText('No mount estimates in this result.')
            return
        lo, hi = min(ys + [0.0]), max(ys + [0.0])
        pad = max(0.05, 0.1 * (hi - lo))
        self.chart.xname, self.chart.yunit = 'episode' if doc.get('time_unit') != 'days' else 'day', '°'
        self.chart.set_data(series, (min(xs), max(xs)), (lo - pad, hi + pad), vlines=vlines,
                            xlabel='episode' if doc.get('time_unit') != 'days' else 'days', ylabel='rotation (°)')
        self.chart_note.setText('Each camera\'s mount rotation (its largest axis, body frame) from its first estimate; '
                                'vertical lines are the steps the classifier found (ported from Vantage).')

    # ---------------------------------------------------------------- suggestion
    def suggestion(self):
        s = (self.doc or {}).get('summary') or {}
        drop = s.get('drop_or_rerecord') or []
        uid = self._current_uid()
        if not drop or uid is None:
            return None
        e = next((d for d in self._datasets() if d['uid'] == uid), None)
        if e is not None and sorted(e.get('exclude') or []) == sorted(drop):
            return None
        return drop

    def apply(self):
        drop = self.suggestion()
        uid = self._current_uid()
        if drop is None:
            return None

        def fn(p):
            p.data = dict(p.data)
            p.data['datasets'] = [dict(d, exclude=list(drop)) if d['uid'] == uid else d
                                  for d in p.data.get('datasets') or []]
        self.ctx.store.edit('data', fn, f'Dataset audit · {len(drop)} episodes marked to drop or re-record')
        self.suggestion_ready(False)
        return drop

    def preview(self):
        drop = self.suggestion()
        if self.ctx.window is not None:
            self.ctx.window.msg('Apply would mark episodes ' + ', '.join(map(str, drop)) + ' to drop or re-record in '
                                'the dataset\'s entry.' if drop else 'Nothing to mark.', 8000)

    def explain(self):
        box = QMessageBox(self)
        box.setWindowTitle('How the dataset is audited')
        box.setText(
            'Every episode is checked from its own data: the camera timestamps (drops, USB gaps, frame rate, jitter), '
            'the pixels (duplicates, exposure steps, blur against the neighbouring frames, the gripper inside the '
            'plan\'s gripper mask), and the joint states (the camera offset that lines the image motion up with the '
            'plan\'s predicted motion, by rank correlation; the action lag). Over the campaign each camera\'s mount '
            'is estimated per episode (a board, the link silhouette, or registration of the static image) and the '
            'series is explained by steps against a gradual drift (model selection by BIC, ported from Vantage).\n\n'
            'Every limit is a DexVantage default you can change below. Known limits: LeRobot stores camera times on '
            'the nominal grid, so its drops show as duplicates; the offset is resolved to about ±12 ms on static '
            'cameras and poorly on cameras on the robot; drifts under several thousand ppm within an episode are '
            'not resolved.')
        box.setModal(False)
        box.show()

    # ---------------------------------------------------------------- exports
    def export_report(self, folder: str | None = None):
        from ...core.audit import report as RP
        if self.doc is None:
            if self.ctx.window is not None:
                self.ctx.window.msg('Run the audit first.', 5000)
            return None
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export the audit report into a folder')
            if not folder:
                return None
        try:
            os.makedirs(folder, exist_ok=True)
            j = RP.write_json(self.doc, self.ctx.store.project, os.path.join(folder, 'audit.json'))
            h = RP.write_html(self.doc, self.ctx.store.project, os.path.join(folder, 'audit.html'))
        except OSError as e:
            if self.ctx.window is not None:
                self.ctx.window.msg(f'Could not write the report into {os.path.basename(os.path.normpath(folder))}: '
                                    f'{(e.strerror or str(e)).lower()}. Choose another folder.', 8000)
            return None
        if self.ctx.window is not None:
            self.ctx.window.msg('Wrote audit.json and audit.html', 5000)
        return j, h

    def export_fiftyone(self, folder: str | None = None, media: str = 'clips'):
        from ...core.audit import report as RP
        if self.doc is None:
            if self.ctx.window is not None:
                self.ctx.window.msg('Run the audit first.', 5000)
            return None
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, 'Export to FiftyOne: choose a new or empty folder')
            if not folder:
                return None
        doc, snap = self.doc, self.ctx.store.project.snapshot()
        try:
            path = AU.dataset_path(snap)
        except Exception as e:                          # noqa: BLE001
            if self.ctx.window is not None:
                self.ctx.window.msg(str(e), 8000)
            return None

        def job(progress, cancel):
            try:
                return RP.export_fiftyone(doc, path, folder, snap, media, progress, cancel)
            except OSError as e:                         # no write access, a full disk: a sentence, not errno
                raise OSError(f'DexVantage cannot write into the folder {os.path.basename(os.path.normpath(folder))} '
                              f'({(e.strerror or str(e)).lower()}): choose another folder') from None
        win = self.ctx.window
        if win is not None and hasattr(win, 'run_job'):
            win.run_job('audit_fiftyone', 'Exporting to FiftyOne', job,
                        lambda r: win.msg(f'FiftyOne dataset written: {r["samples"]} samples, {r["frames"]:,} frames. '
                                          'Load it with fiftyone.Dataset.from_dir(folder, dataset_type='
                                          'fiftyone.types.FiftyOneDataset).', 10000),
                        lambda m: win.msg(f'The FiftyOne export stopped: {m.splitlines()[0]}', 10000))
            return folder
        return job(lambda *_: None, lambda: False)

    # ---------------------------------------------------------------- overlays
    def _camera_status(self) -> dict:
        """{sensor uid: worst status of the audit's camera} over the episodes."""
        out = {}
        rank = {'good': 0, 'warning': 1, 'serious': 2, 'critical': 3}
        for e in (self.doc or {}).get('episodes') or []:
            for key, c in e['cameras'].items():
                uid = c.get('sensor_uid')
                if not uid:
                    continue
                bad = [r for r in e['requirements'] if r['camera'] == key and not r['pass']]
                sev = 'good' if not bad else ('warning' if all(r['id'] in AE.FIXABLE for r in bad) else 'critical')
                cur = out.get(uid)
                if cur is None or rank[sev] > rank[cur]:
                    out[uid] = sev
        return out

    def overlays(self, vp) -> list:
        from ...core import camrender as CR
        from ...core import library as LB
        from ...core.studies import coverage as CV
        from ..viewport3d import LineDraw
        if self.doc is None:
            return []
        out = []
        p = self.ctx.store.project
        for uid, sev in self._camera_status().items():
            s = p.variant.sensor(uid)
            if s is None:
                continue
            try:
                im, _w = CV.pick_imager(s, LB.sensor_imagers(s, CV.sensor_mode(s, {})), {})
                pose = CR.optical_pose(p, s, im, self.ctx.store.q)
            except Exception:                           # noqa: BLE001 — a camera that cannot be posed is skipped
                continue
            o = np.asarray(pose.position, float)
            from ...core.frames import quat_to_matrix
            z = quat_to_matrix(pose.quaternion)[:, 2]
            out.append(LineDraw(np.array([o, o + 0.15 * z]), QColor(STATUS[sev]), width=3.0, depth=False))
        return out

    def camview_overlays(self, cam_model, pose) -> list:
        """For a camera whose mount moved (registration or board), the planned image border as it sits after the
        move (the last episode's estimate)."""
        from ...core.audit.campaign import _rotvec_R
        from ...core.camrender import image_border
        if self.doc is None:
            return []
        cv = getattr(self.ctx.window, 'camview', None) if self.ctx.window is not None else None
        uid = getattr(cv, 'sensor_uid', None) or getattr(getattr(cv, 'view', None), 'sensor_uid', None)
        for key, m in (self.doc.get('mounts') or {}).items():
            if m.get('sensor_uid') != uid or m.get('kind') not in ('slip', 'excursion', 'gradual'):
                continue
            S = m.get('episodes') or []
            if len(S) < 2:
                return []
            try:
                from ...core import library as LB
                from ...core.studies import coverage as CV
                s = self.ctx.store.project.variant.sensor(uid)
                im, _w = CV.pick_imager(s, LB.sensor_imagers(s, CV.sensor_mode(s, {})), {})
                R_so = im.pose_in_body.matrix()[:3, :3]
                w = np.radians(np.subtract([v or 0.0 for v in S[-1]['rotation_deg']],
                                           [v or 0.0 for v in S[0]['rotation_deg']]))
                A = _rotvec_R(R_so.T @ w)
                rays = image_border(cam_model)
                uv, ok = cam_model.project(rays @ A)
                uv = uv[np.isfinite(uv).all(axis=1)]
                if len(uv) < 4:
                    return []
                return [([uv], QColor(STATUS['serious']), f'planned view after the mount moved (episode '
                                                          f'{S[-1]["episode"]})')]
            except Exception:                           # noqa: BLE001 — no overlay rather than a broken view
                traceback.print_exc()
                return []
        return []

    def _redraw(self):
        win = self.ctx.window
        if win is not None and hasattr(win, 'viewport'):
            win.viewport.update()
        cv = getattr(win, 'camview', None) if win is not None else None
        if cv is not None and hasattr(cv, 'view'):
            cv.view.refresh_overlays()


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


register('audit', 'Dataset audit', AuditPage, menu='Data', icon='dataset', action_id='audit', workspace='audit',
         order=80)
