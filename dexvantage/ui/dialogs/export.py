"""File ▸ Export… (and Export ▸ Export to MuJoCo, Isaac, ROS 2, LeRobot…): pick the formats (each with its
description), their options, the pose and the output folder; the export runs as a window job (core/exporters.export)
and the dialog then shows the manifest: each format's status, files and notes.

    open_dialog(window) -> ExportDialog
    run(window, folder, formats, options, on_done=None) -> bool     the window job (also for scripts and tests)

Options are each exporter's own (its module docstring); the dialog offers the common ones as controls and the rest as
JSON per format. The pose is the home pose of every robot, or the pose on screen (the 3D view's joint state).
"""
from __future__ import annotations

import json
import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLineEdit, QPlainTextEdit,
                             QScrollArea, QVBoxLayout, QWidget)

from ...core import exporters as X
from ..widgets import button, label
from .base import VDialog

# the common options offered as controls: format id -> [(key, title, kind, default, choices)]
CONTROLS = {
    'mjcf': [('mode', 'What', 'choice', 'both', (('Snippets and the full scene', 'both'),
                                                  ('Camera and site snippets', 'snippets'),
                                                  ('The full scene', 'scene')))],
    'urdf': [('package', 'ROS package for mesh paths (empty: relative)', 'text', '', None)],
    'opencv_yaml': [('imagers', 'Imagers', 'choice', 'all', (('Every imager', 'all'),
                                                            ('Each sensor\'s main imager', 'primary')))],
    'masks': [('atlas', 'Mask atlas over the grasp set', 'bool', True, None),
              ('depth', 'Depth-validity masks', 'bool', True, None)],
    'brackets': [('step', 'Also STEP (needs gmsh)', 'bool', False, None)],
    'fiftyone': [('dataset', 'Dataset to audit (folder)', 'text', '', None)],
    'report': [('run_missing', 'Also run the studies not computed yet', 'bool', True, None)],
}
POSED = ('mjcf', 'urdf', 'opencv_yaml', 'masks', 'plucker', 'isaac')    # the formats whose output depends on the pose


class _Row(QWidget):
    def __init__(self, info, parent=None):
        super().__init__(parent)
        self.info = info
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 6)
        lay.setSpacing(3)
        self.check = QCheckBox(info.title + (' (needs a robot)' if info.needs_robot else ''))
        lay.addWidget(self.check)
        if info.description:
            d = label(info.description, muted=True, small=True, wrap=True)
            d.setContentsMargins(22, 0, 0, 0)
            lay.addWidget(d)
        self.controls = {}
        g = QGridLayout()
        g.setContentsMargins(22, 0, 0, 0)
        g.setHorizontalSpacing(8)
        r = 0
        for key, title, kind, default, choices in CONTROLS.get(info.id, ()):
            if kind == 'bool':
                w = QCheckBox(title)
                w.setChecked(bool(default))
                g.addWidget(w, r, 0, 1, 2)
            elif kind == 'choice':
                w = QComboBox()
                for text, v in choices:
                    w.addItem(text, v)
                w.setCurrentIndex(next(i for i, (_t, v) in enumerate(choices) if v == default))
                g.addWidget(label(title, small=True), r, 0)
                g.addWidget(w, r, 1)
            else:
                w = QLineEdit(str(default))
                g.addWidget(label(title, small=True), r, 0)
                g.addWidget(w, r, 1)
            self.controls[key] = (kind, w)
            r += 1
        self.extra = QLineEdit()
        self.extra.setPlaceholderText('More options as JSON, e.g. {"scale": 0.5}')
        g.addWidget(label('Options', small=True), r, 0)
        g.addWidget(self.extra, r, 1)
        lay.addLayout(g)
        self.note = label('', small=True, wrap=True)
        self.note.setContentsMargins(22, 0, 0, 0)
        self.note.hide()
        lay.addWidget(self.note)

    def set_note(self, lines: list):
        """Statements about this format on the open project (shown under its options; hidden when none)."""
        self.note.setText('\n'.join(lines))
        self.note.setVisible(bool(lines))

    def options(self) -> dict:
        """The options of this format; raises ValueError (said in words) when the JSON is not an object."""
        out = {}
        for key, (kind, w) in self.controls.items():
            if kind == 'bool':
                out[key] = w.isChecked()
            elif kind == 'choice':
                out[key] = w.currentData()
            else:
                t = w.text().strip()
                if t:
                    out[key] = t
        t = self.extra.text().strip()
        if t:
            try:
                more = json.loads(t)
            except ValueError as e:
                raise ValueError(f'The options of {self.info.title} are not valid JSON ({e.msg}).') from None
            if not isinstance(more, dict):
                raise ValueError(f'The options of {self.info.title} must be a JSON object, e.g. {{"scale": 0.5}}.')
            out.update(more)
        return out


class ExportDialog(VDialog):
    def __init__(self, window):
        super().__init__(window, 'Export', 'files for MuJoCo, ROS 2, OpenCV, Kalibr, training', (760, 700))
        self.window_ = window
        X.load_exporters()
        lay = QVBoxLayout(self.body)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)
        lay.addWidget(label('Each format writes into its own subfolder of the output folder; export_manifest.json '
                            'lists every file with its SHA-256 and the project\'s provenance. A format that cannot run '
                            'on this project is refused with the reason, and the others are still written.',
                            wrap=True, muted=True))
        inner = QWidget()
        il = QVBoxLayout(inner)
        il.setContentsMargins(0, 0, 0, 0)
        il.setSpacing(2)
        self.rows = {}
        for fid in sorted(X.EXPORTERS, key=lambda f: X.EXPORTERS[f].title.lower()):
            if fid.startswith('_'):
                continue
            row = _Row(X.EXPORTERS[fid])
            self.rows[fid] = row
            il.addWidget(row)
        il.addStretch(1)
        self._format_notes()
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(inner)
        lay.addWidget(sc, 3)
        g = QGridLayout()
        g.setHorizontalSpacing(8)
        self.pose = QComboBox()
        self.pose.addItem('The home pose of every robot', 'home')
        self.pose.addItem('The pose on screen', 'screen')
        g.addWidget(label('Pose'), 0, 0)
        g.addWidget(self.pose, 0, 1)
        g.addWidget(label('Output folder'), 1, 0)
        rowf = QWidget()
        h = QHBoxLayout(rowf)
        h.setContentsMargins(0, 0, 0, 0)
        self.folder = QLineEdit()
        self.folder.setPlaceholderText('A folder for the exported files')
        h.addWidget(self.folder, 1)
        b = button('Browse…')
        b.clicked.connect(self._browse)
        h.addWidget(b)
        g.addWidget(rowf, 1, 1)
        lay.addLayout(g)
        lay.addWidget(label('Manifest', bold=True))
        self.result = QPlainTextEdit()
        self.result.setReadOnly(True)
        self.result.setPlaceholderText('After the export: each format, its files and its notes.')
        lay.addWidget(self.result, 2)
        self.hint.setText('The pose matters for cameras on a robot link (their world poses and masks).')
        c = button('Close')
        c.clicked.connect(self.reject)
        self.go = button('Export', 'primary', 'export')
        self.go.clicked.connect(self.start)
        self.buttons.addWidget(c)
        self.buttons.addWidget(self.go)

    def _format_notes(self):
        """Per-camera statements the Isaac export makes on this project (an off-centre principal point: where it
        reaches an Isaac render), before anything is exported; plain sentences, never a traceback."""
        row = self.rows.get('isaac')
        if row is None:
            return
        try:
            from ...core.exporters import isaac as XI
            lines = XI.principal_point_lines(self.window_.ctx.store.project)
        except Exception:                               # noqa: BLE001 — a courtesy line: logged, not shown
            import traceback
            traceback.print_exc()
            lines = []
        row.set_note(lines)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, 'Folder for the exported files')
        if d:
            self.folder.setText(d)

    def chosen(self) -> list:
        return [fid for fid, r in self.rows.items() if r.check.isChecked()]

    def set_chosen(self, fids):
        for fid, r in self.rows.items():
            r.check.setChecked(fid in fids)

    def options(self) -> dict:
        out = {fid: self.rows[fid].options() for fid in self.chosen()}
        if self.pose.currentData() == 'screen':
            q = {uid: [float(x) for x in v] for uid, v in (self.window_.ctx.store.q or {}).items()}
            for fid in out:
                if fid in POSED:
                    out[fid].setdefault('q', q)
        for rid in [f for f in ('report', 'paper_kit') if f in out]:
            # the report (and the paper kit) reuse the results the window has already computed for this revision of
            # the design (a study computed for an earlier revision is run again); run_missing adds every other study
            rep = out[rid]
            run_missing = bool(rep.pop('run_missing', True))
            if 'results' not in rep and not rep.get('results_dir'):
                an = getattr(self.window_.ctx, 'analysis', None)
                rev = self.window_.ctx.store.revision
                rep['results'] = {sid: {k: v for k, v in d.items() if k != 'revision'}
                                  for sid, d in dict(getattr(an, 'results', {}) or {}).items()
                                  if isinstance(d, dict) and d.get('revision', rev) == rev}
                if run_missing:
                    rep.setdefault('studies', 'all')
        return out

    def start(self) -> bool:
        fmts = self.chosen()
        if not fmts:
            self.hint.setText('Tick at least one format.')
            return False
        folder = self.folder.text().strip()
        if not folder:
            self.hint.setText('Choose an output folder first.')
            return False
        if os.path.exists(folder) and not os.path.isdir(folder):
            self.hint.setText('That path is a file: choose a folder.')
            return False
        try:
            opts = self.options()
        except ValueError as e:
            self.hint.setText(str(e))
            return False
        self.result.setPlainText('Exporting…')
        started = run(self.window_, folder, fmts, opts, on_done=self.show_manifest)
        if started:
            self.go.setEnabled(False)
            self.hint.setText('Exporting in the background (progress in the status bar).')
        else:
            self.hint.setText('An export is already running: wait for it to finish.')
        return started

    def show_manifest(self, man: dict | None, error: str = ''):
        self.go.setEnabled(True)
        if man is None:
            self.result.setPlainText(error)
            self.hint.setText('The export stopped.')
            return
        self.result.setPlainText(manifest_text(man))
        n_ok = sum(1 for r in man['formats'].values() if r['status'] == 'ok')
        self.hint.setText(f'{n_ok} of {len(man["formats"])} formats written; {len(man["files"])} files.')


def manifest_text(man: dict) -> str:
    """The manifest in plain lines: per format its status, file count, notes or reason."""
    lines = []
    for fid, r in man['formats'].items():
        if r['status'] == 'ok':
            lines.append(f'{r["title"]} ({fid}): {len(r["files"])} file{"" if len(r["files"]) == 1 else "s"}')
            lines += [f'  · {n}' for n in r['notes']]
        else:
            lines.append(f'{r["title"]} ({fid}): refused. {r.get("reason", "")}')
        lines.append('')
    if man.get('folder'):
        lines.append(f'Folder: {man["folder"]}')
    return '\n'.join(lines).rstrip() + '\n'


def run(window, folder: str, formats: list, options: dict, on_done=None) -> bool:
    """Export a snapshot of the project on a window job; on_done(manifest) or on_done(None, message) here."""
    snap = window.ctx.store.project.snapshot()

    def job(progress, cancel):
        from ...core.jobs import Ctx, StudyError
        try:
            man = X.export(snap, folder, formats, options, Ctx(progress=progress, cancel=cancel))
        except StudyError as e:
            return {'error': str(e)}
        man['folder'] = os.path.basename(os.path.normpath(folder))
        return {'manifest': man}

    def done(res):
        if 'error' in res:
            window.msg(f'Could not export: {res["error"]}', 10000)
            if on_done is not None:
                on_done(None, res['error'])
            return
        man = res['manifest']
        n_ok = sum(1 for r in man['formats'].values() if r['status'] == 'ok')
        window.msg(f'Exported {n_ok} of {len(man["formats"])} formats into {man["folder"]}', 8000)
        try:
            window.ctx.store.project.add_log('export', f'Exported {", ".join(formats)} into {man["folder"]}')
        except Exception:                               # noqa: BLE001 — the log line is a courtesy
            pass
        if on_done is not None:
            on_done(man)

    def failed(msg):                                     # the window logged the traceback
        text = 'The export stopped with an unexpected error (details in the log).'
        window.msg(text, 10000)
        if on_done is not None:
            on_done(None, text)
    return bool(window.run_job('export', 'Exporting', job, done, failed))


def open_dialog(window):
    d = ExportDialog(window)
    d.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
    d.fit_screen()
    d.show()
    return d
