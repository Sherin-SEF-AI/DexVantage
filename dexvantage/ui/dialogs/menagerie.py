"""Robot ▸ Import from MuJoCo Menagerie…: pick a model, read its own licence, accept it, import it (spec §1.2, §9.4).

The list comes from core/robot/menagerie.index() (GitHub, or the built-in list of folder names when offline, which
the dialog says). "Download and show the licence" fetches the model folder and shows its LICENSE text; Import stays
disabled until the user ticks "I accept the licence of this model" for that model. The import is one undo step over
the project's robots and sources, like Robot ▸ Import robot…, and records the licence on the robot.

Network work runs on a background thread; tests pass an `opener` and `background=False` so nothing touches the
network or waits.
"""
from __future__ import annotations

import threading
import traceback

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLineEdit, QListWidget, QPlainTextEdit, QVBoxLayout,
                             QWidget)

from ...core import project as P
from ...core.robot import menagerie as MG
from .. import icons
from ..theme import mono
from ..widgets import StatusLine, button, label
from .base import VDialog


class _Relay(QObject):
    """Results from the worker thread, delivered on the GUI thread (queued signals)."""
    done = pyqtSignal(str, object)
    failed = pyqtSignal(str, str)
    progress = pyqtSignal(float, str)


class MenagerieDialog(VDialog):
    def __init__(self, window, opener=None, dest=None, background=True, parent=None):
        super().__init__(parent or window, 'Import from MuJoCo Menagerie', 'each model has its own licence',
                         (900, 600))
        self.win = window
        self.opener = opener
        self.dest = dest
        self.background = background
        self.dl: MG.MenagerieDownload | None = None
        self.robot = None
        self._busy = False
        self._relay = _Relay()
        self._relay.done.connect(self._on_done)
        self._relay.failed.connect(self._on_failed)
        self._relay.progress.connect(lambda f, t: self.status.set('', f'Downloading {f * 100:.0f} % · {t}'))

        lay = QHBoxLayout(self.body)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(14)
        left = QVBoxLayout()
        self.filter = QLineEdit()
        self.filter.setPlaceholderText('Filter models')
        self.filter.addAction(icons.icon('search', 12), QLineEdit.ActionPosition.LeadingPosition)
        self.filter.textChanged.connect(self._apply_filter)
        left.addWidget(self.filter)
        self.view = QListWidget()
        self.view.currentTextChanged.connect(self._on_pick)
        left.addWidget(self.view, 1)
        self.index_note = label('', muted=True, small=True, wrap=True)
        left.addWidget(self.index_note)
        lw = QWidget()
        lw.setLayout(left)
        lw.setFixedWidth(280)
        lay.addWidget(lw)

        right = QVBoxLayout()
        right.setSpacing(8)
        self.title = label('Pick a model', bold=True)
        right.addWidget(self.title)
        row = QHBoxLayout()
        self.fetch = button('Download and show the licence', 'primary', 'import',
                            tip='Downloads the model folder and shows its own LICENSE before anything is imported')
        self.fetch.clicked.connect(self.fetch_selected)
        self.fetch.setEnabled(False)
        row.addWidget(self.fetch)
        row.addStretch(1)
        right.addLayout(row)
        self.status = StatusLine('', 'Menagerie licences differ per model: read the licence before importing.')
        right.addWidget(self.status)
        self.lic_name = label('Licence: —', mono_=True)
        right.addWidget(self.lic_name)
        self.lic_text = QPlainTextEdit()
        self.lic_text.setReadOnly(True)
        self.lic_text.setFont(mono(11))
        self.lic_text.setPlaceholderText('The model\'s LICENSE file appears here after the download.')
        right.addWidget(self.lic_text, 1)
        frow = QHBoxLayout()
        frow.addWidget(label('Model file', ink2=True))
        self.file = QComboBox()
        self.file.setEnabled(False)
        frow.addWidget(self.file, 1)
        right.addLayout(frow)
        self.accept_box = QCheckBox('I accept the licence of this model')
        self.accept_box.setEnabled(False)
        self.accept_box.toggled.connect(self._sync)
        right.addWidget(self.accept_box)
        lay.addLayout(right, 1)

        self.hint.setText('Nothing is imported until you accept the licence.')
        c = button('Cancel', 'ghost')
        c.clicked.connect(self.reject)
        self.import_btn = button('Import', 'primary')
        self.import_btn.clicked.connect(self.import_selected)
        self.import_btn.setEnabled(False)
        self.buttons.addWidget(c)
        self.buttons.addWidget(self.import_btn)
        self.models: list = []

    # ------------------------------------------------------------------ work on a thread
    def _run(self, kind, fn):
        self._busy = True
        self._sync()
        if not self.background:
            try:
                self._on_done(kind, fn(None))
            except Exception as e:                          # noqa: BLE001 — explained
                self._on_failed(kind, self._message(e))
            return

        def work():
            try:
                self._relay.done.emit(kind, fn(lambda f, t: self._relay.progress.emit(f, t)))
            except Exception as e:                          # noqa: BLE001 — explained
                self._relay.failed.emit(kind, self._message(e))
        threading.Thread(target=work, daemon=True).start()

    @staticmethod
    def _message(e) -> str:
        if isinstance(e, (MG.MenagerieError, P.ProjectFormatError)):
            return str(e)
        traceback.print_exc()
        return f'Something went wrong ({type(e).__name__}); details are in the log.'

    def _on_done(self, kind, res):
        self._busy = False
        if kind == 'index':
            self.models = list(res['models'])
            self.index_note.setText(res['note'])
            self._apply_filter(self.filter.text())
        elif kind == 'download':
            self._show_download(res)
        self._sync()

    def _on_failed(self, kind, text):
        self._busy = False
        self.status.set('critical', f'Failed: {text}')
        self._sync()

    # ------------------------------------------------------------------ the list
    def load_index(self):
        self.index_note.setText('Asking GitHub for the list of models…')
        self._run('index', lambda _p: MG.index(self.opener))

    def _apply_filter(self, text):
        cur = self.view.currentItem().text() if self.view.currentItem() else ''
        t = str(text or '').strip().lower()
        self.view.blockSignals(True)
        self.view.clear()
        for m in self.models:
            if not t or t in m.lower():
                self.view.addItem(m)
        self.view.blockSignals(False)
        hits = self.view.findItems(cur, Qt.MatchFlag.MatchExactly) if cur else []
        if hits:
            self.view.setCurrentItem(hits[0])

    def select_model(self, name: str) -> bool:
        hits = self.view.findItems(name, Qt.MatchFlag.MatchExactly)
        if hits:
            self.view.setCurrentItem(hits[0])
        return bool(hits)

    def _on_pick(self, name):
        if self.dl is not None and self.dl.model == name:
            return
        self.dl = None
        self.title.setText(name or 'Pick a model')
        self.lic_name.setText('Licence: —')
        self.lic_text.setPlainText('')
        self.file.clear()
        self.accept_box.setChecked(False)
        self.status.set('', 'Menagerie licences differ per model: read the licence before importing.')
        self._sync()

    # ------------------------------------------------------------------ download, licence, import
    def fetch_selected(self):
        item = self.view.currentItem()
        if item is None or self._busy:
            return
        name = item.text()
        self.status.set('', f'Downloading {name}…')
        self._run('download', lambda prog: MG.download(name, self.dest, self.opener, prog))

    def _show_download(self, dl: MG.MenagerieDownload):
        cur = self.view.currentItem()
        if cur is None or cur.text() != dl.model:
            return                                         # the user moved on while it downloaded
        self.dl = dl
        self.lic_name.setText(f'Licence: {dl.license_name}   ·   Menagerie commit {dl.commit[:12]}')
        self.lic_text.setPlainText(dl.license_text)
        self.file.clear()
        self.file.addItems(dl.mjcf)
        self.accept_box.setChecked(False)
        if dl.noncommercial:
            self.status.set('serious', 'Restricted: this licence does not allow commercial use. Import it only if '
                                       'your use is non-commercial.')
        elif dl.license_name.startswith('see'):
            self.status.set('warning', 'Unrecognised: DexVantage does not recognise this licence text. Read it in '
                                       'full before accepting.')
        else:
            self.status.set('good', f'Downloaded{" (already on this computer)" if dl.cached else ""}: '
                                    f'{len(dl.files)} files. Read the licence, then accept it to import.')
        if not dl.mjcf:
            self.status.set('critical', f'Not importable: {dl.model} has no MuJoCo model file (.xml).')
        self._sync()

    def _sync(self, *_):
        has = self.view.currentItem() is not None
        self.fetch.setEnabled(has and not self._busy)
        ready = self.dl is not None and bool(self.dl.mjcf) and not self._busy
        self.file.setEnabled(ready)
        self.accept_box.setEnabled(ready)
        if not ready and self.accept_box.isChecked():
            self.accept_box.setChecked(False)
        self.import_btn.setEnabled(ready and self.accept_box.isChecked())

    def import_selected(self) -> bool:
        """Import the downloaded model (only after the licence was accepted) as one undo step."""
        if self.dl is None or not self.accept_box.isChecked():
            self.status.set('warning', 'Not imported: accept the model\'s licence first.')
            return False
        try:
            robot, sources = MG.import_download(self.dl, self.file.currentText() or None)
        except Exception as e:                              # noqa: BLE001 — explained
            self.status.set('critical', f'Not imported: {self._message(e)}')
            return False

        def fn(p):
            robot.primary = not p.robots
            p.robots.append(robot)
            p.sources.update(sources)
        st = self.win.store
        st.edit(['robots', 'sources'], fn, f'Imported {robot.name} from MuJoCo Menagerie ({self.dl.license_name})')
        st.select(('robot', robot.uid))
        self.robot = robot
        if hasattr(self.win, 'frame_all'):
            self.win.frame_all()
        if hasattr(self.win, 'msg'):
            self.win.msg(f'Imported {robot.name} from MuJoCo Menagerie; licence {self.dl.license_name} recorded')
        self.accept()
        return True


def open_dialog(window) -> bool:
    """Show the dialog for the window (Robot ▸ Import from MuJoCo Menagerie…); True when a robot was imported."""
    d = MenagerieDialog(window)
    d.load_index()
    d.exec()
    return d.robot is not None
