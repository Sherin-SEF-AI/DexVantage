"""Simulate ▸ Generate synthetic episodes…: a small dialog (episodes, frames, LeRobot version, image size, seed, the
faults to inject, the folder) that runs core/sim/episodes.generate as a window job with progress."""
from __future__ import annotations

import os

from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLineEdit, QVBoxLayout,
                             QWidget)

from ...core.sim import episodes as SE
from ..widgets import ScrubSpinBox, button, label
from .base import VDialog

SIZES = (('Up to 640 px wide (default)', 640), ('Up to 320 px wide', 320), ('Up to 160 px wide', 160),
         ('The stream size', 100000))


class SynthEpisodesDialog(VDialog):
    def __init__(self, window):
        super().__init__(window, 'Generate synthetic episodes', 'LeRobot dataset with known faults', (620, 560),
                         resizable=True)
        self.window_ = window
        p = window.ctx.store.project
        try:
            st = SE.settings_of(p)
        except SE.SynthError:                           # a bad stored setting: the dialog starts from the defaults
            st = SE.settings_of(None)
        lay = QVBoxLayout(self.body)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)
        lay.addWidget(label('The robot follows the project\'s trajectories (or approaches its grasps) and every camera '
                            'is rendered in MuJoCo through its own lens. The faults you tick are injected at seeded, '
                            'known places and written to dexvantage_truth.json beside the dataset, so the auditor can '
                            'be tested against them.', wrap=True, muted=True))
        g = QGridLayout()
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(6)
        self.n_episodes = self._spin(st['episodes'], 1, 10000)
        self.n_frames = self._spin(st['frames'] or 0, 0, 100000)
        self.seed = self._spin(st['seed'], 0, 2 ** 31 - 1)
        self.version = QComboBox()
        for v in ('v3.0', 'v2.1'):
            self.version.addItem(f'LeRobot {v}', v)
        self.version.setCurrentIndex(0 if st['version'] == 'v3.0' else 1)
        self.size = QComboBox()
        for text, w in SIZES:
            self.size.addItem(text, w)
        self.size.setCurrentIndex(next((i for i, (_t, w) in enumerate(SIZES) if w == st['max_width']), 0))
        rows = (('Episodes', self.n_episodes), ('Rows per episode (0: the motion\'s own length)', self.n_frames),
                ('Format', self.version), ('Images', self.size), ('Seed', self.seed))
        for r, (t, w) in enumerate(rows):
            g.addWidget(label(t), r, 0)
            g.addWidget(w, r, 1)
        lay.addLayout(g)
        lay.addWidget(label('Faults to inject', bold=True))
        fg = QGridLayout()
        fg.setHorizontalSpacing(16)
        self.faults = {}
        for i, k in enumerate(SE.FAULT_KINDS):
            cb = QCheckBox(SE.FAULT_TITLES[k])
            cb.setChecked(bool(st['faults'][k]['enabled']))
            self.faults[k] = cb
            fg.addWidget(cb, i // 2, i % 2)
        lay.addLayout(fg)
        lay.addWidget(label('Folder (new or empty)', bold=True))
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        self.folder = QLineEdit()
        self.folder.setPlaceholderText('Choose a new or empty folder for the dataset')
        h.addWidget(self.folder, 1)
        b = button('Browse…')
        b.clicked.connect(self._browse)
        h.addWidget(b)
        lay.addWidget(row)
        lay.addStretch(1)
        self.hint.setText('The same seed places the faults at the same frames.')
        c = button('Cancel')
        c.clicked.connect(self.reject)
        self.go = button('Generate', 'primary')
        self.go.clicked.connect(self.start)
        self.buttons.addWidget(c)
        self.buttons.addWidget(self.go)

    @staticmethod
    def _spin(value, lo, hi):
        s = ScrubSpinBox('', 'number', 0, width=110)
        s.setRange(lo, hi)
        s.setValue(int(value))
        return s

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, 'Folder for the synthetic dataset')
        if d:
            self.folder.setText(d)

    def settings(self) -> dict:
        return {'episodes': int(self.n_episodes.value()), 'frames': int(self.n_frames.value()) or None,
                'version': self.version.currentData(), 'max_width': int(self.size.currentData()),
                'seed': int(self.seed.value()), 'faults': [k for k, cb in self.faults.items() if cb.isChecked()]}

    def start(self):
        """Check the folder, then generate on a window job; the dialog closes when the job starts."""
        folder = self.folder.text().strip()
        if not folder:
            self.hint.setText('Choose a folder for the dataset first.')
            return False
        if os.path.exists(folder) and (not os.path.isdir(folder) or os.listdir(folder)):
            self.hint.setText('That folder is not empty: choose a new or empty folder.')
            return False
        if not self.window_.ctx.store.project.robots:
            self.hint.setText('There is no robot to record: import one or open a sample.')
            return False
        started = run(self.window_, folder, self.settings())
        if started:
            self.accept()
        else:
            self.hint.setText('Synthetic episodes are already being generated: wait for that job to finish.')
        return started


def run(window, folder: str, settings: dict) -> bool:
    """Generate on a window job (progress in the status bar, Cancel stops it); a message when done or failed."""
    snap = window.ctx.store.project.snapshot()

    def job(progress, cancel):
        try:
            return SE.generate(snap, folder, settings, progress=progress, cancel=cancel)
        except OSError as e:                             # no write access, a full disk: a sentence, not errno
            raise OSError(f'DexVantage cannot write into the folder {os.path.basename(os.path.normpath(folder))} '
                          f'({(e.strerror or str(e)).lower()}): choose another folder') from None

    def done(s):
        window.msg(SE.summary_text({k: v for k, v in s.items() if k != 'truth'}) + f' Folder: '
                   f'{os.path.basename(s["root"])}.', 12000)

    def failed(msg):
        window.msg(f'The synthetic episodes could not be generated: {msg.splitlines()[0]}', 12000)
    return bool(window.run_job('synth_episodes', 'Generating synthetic episodes', job, done, failed))


def open_dialog(window):
    """Simulate ▸ Generate synthetic episodes… from the main window."""
    d = SynthEpisodesDialog(window)
    d.show()
    return d
