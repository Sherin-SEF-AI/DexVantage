"""A small dialog that chooses a recorded dataset to read from (LeRobot v2.1 / v3.0 folder, a ROS 2 bag folder or
.mcap / .db3 file, an RLDS folder) and how to read it: which episodes, every Nth row or frame, the joints' unit, and
for a hand-eye capture the camera, its stream, the time offset and whether frames taken while the robot moves are
used. It only collects choices; the pages read the dataset on a window job (core/grasps.from_dataset,
core/calib/capture.capture_dataset)."""
from __future__ import annotations

import os
import re

from PyQt6.QtWidgets import QCheckBox, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QLineEdit, QVBoxLayout, \
    QWidget

from ..widgets import ScrubSpinBox, SegmentedControl, button, label
from .base import VDialog

BAG_FILES = 'Datasets and bags (*.mcap *.db3 *.tfrecord *.tfrecord-* metadata.yaml info.json);;All files (*)'


def parse_episodes(text: str):
    """None for all episodes ('', 'all'), else the sorted episode indices of '0, 2-4, 7'; ValueError in words."""
    t = (text or '').strip().lower()
    if t in ('', 'all', '*'):
        return None
    out: set = set()
    for part in re.split(r'[,\s;]+', t):
        if not part:
            continue
        m = re.fullmatch(r'(\d+)\s*(?:-|–|\.\.)\s*(\d+)', part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if b < a:
                raise ValueError(f'"{part}" runs backwards: write the lower episode first ({b}-{a}).')
            out.update(range(a, b + 1))
        elif part.isdigit():
            out.add(int(part))
        else:
            raise ValueError(f'"{part}" is not an episode number: write episodes as 0, 2-4, or all.')
    return sorted(out)


def dataset_root(path: str) -> str:
    """The dataset a chosen file stands for: a LeRobot meta/info.json -> its dataset folder, a bag's metadata.yaml ->
    the bag folder; any other path as it is."""
    p = os.path.normpath(path)
    base = os.path.basename(p).lower()
    parent = os.path.dirname(p)
    if base == 'info.json' and os.path.basename(parent) == 'meta':
        return os.path.dirname(parent)
    if base == 'metadata.yaml':
        return parent
    return p


class DatasetSourceDialog(VDialog):
    """kind 'grasps' (the Tasks page) or 'handeye' (the Hand-eye solve page). sensors: [(name, uid)] for 'handeye';
    values() after exec() gives the choices."""

    def __init__(self, parent, kind: str = 'grasps', sensors=(), sensor_uid: str = ''):
        title = 'Grasps from a dataset' if kind == 'grasps' else 'Hand-eye capture from a dataset'
        super().__init__(parent, title, 'LeRobot dataset, ROS 2 bag or RLDS', (600, 420 if kind == 'grasps' else 500),
                         resizable=True)
        self.kind = kind
        lay = QVBoxLayout(self.body)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)
        lay.addWidget(label(
            'The grasps are where the gripper closes in each episode\'s recorded joint states.' if kind == 'grasps' else
            'The frames of the camera\'s stream with the board in view are taken, each with the joint states at its '
            'time (interpolated between the recorded rows).', wrap=True, muted=True))
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        self.path = QLineEdit()
        self.path.setPlaceholderText('A LeRobot dataset folder, a ROS 2 bag folder or .mcap / .db3 file')
        h.addWidget(self.path, 1)
        b_dir = button('Folder…', role='ghost')
        b_dir.clicked.connect(self._browse_dir)
        b_file = button('File…', role='ghost')
        b_file.clicked.connect(self._browse_file)
        h.addWidget(b_dir)
        h.addWidget(b_file)
        lay.addWidget(row)
        g = QGridLayout()
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(6)
        self.episodes = QLineEdit('all')
        self.episodes.setToolTip('The episodes to read: all, or numbers and ranges such as 0, 2-4')
        self.every = ScrubSpinBox('', 'plain', 0, '', step=1.0, width=80)
        self.every.setRange(1, 100000)
        self.every.set_value(1 if kind == 'handeye' else 10)
        self.unit = SegmentedControl([('rad', 'rad'), ('deg', 'deg')], 'rad')
        self.unit.setToolTip('The unit of the recorded joint values (revolute joints; sliding joints are metres)')
        rows = [('Episodes', self.episodes)]
        if kind == 'grasps':
            self.mode = SegmentedControl([('closures', 'Where it closes'), ('every', 'Every N rows')], 'closures')
            self.mode.setToolTip('Where it closes: one grasp each time the gripper closes by 5 mm or more (a default). '
                                 'Every N rows: the gripper frame at every Nth recorded row.')
            self.every.setEnabled(False)
            self.mode.changed.connect(lambda k: self.every.setEnabled(k == 'every'))
            rows += [('Grasps', self.mode), ('Every N rows', self.every)]
        else:
            self.sensor = QComboBox()
            for name, uid in sensors:
                self.sensor.addItem(name, uid)
            k = self.sensor.findData(sensor_uid)
            if k >= 0:
                self.sensor.setCurrentIndex(k)
            self.camera = QLineEdit()
            self.camera.setPlaceholderText('automatic: the stream named like the camera')
            self.camera.setToolTip('The dataset\'s camera key (LeRobot: observation.images.<name>; a bag: the image '
                                   'topic). Leave empty to take the stream whose name matches the camera\'s.')
            self.offset = ScrubSpinBox('', 'plain', 1, 'ms', step=1.0, width=90)
            self.offset.setRange(-10000, 10000)
            self.offset.set_value(0.0)
            self.offset.setToolTip('The joint reading for a frame stamped t is taken at t + offset (default 0: the '
                                   'camera and the joints on one clock)')
            self.moving = QCheckBox('Use frames taken while the robot moves')
            self.moving.setToolTip('Off (default): only frames where the joints move slower than 0.5°/s (a default) '
                                   'are used, since the solve assumes still poses')
            self.every.setToolTip('Look at every Nth frame of the stream')
            rows += [('Camera', self.sensor), ('Camera stream', self.camera), ('Every N frames', self.every),
                     ('Joint time offset', self.offset), ('', self.moving)]
        rows.append(('Joints in', self.unit))
        for r, (t, w) in enumerate(rows):
            g.addWidget(label(t), r, 0)
            g.addWidget(w, r, 1)
        lay.addLayout(g)
        lay.addStretch(1)
        c = button('Cancel')
        c.clicked.connect(self.reject)
        self.go = button('Read', 'primary')
        self.go.clicked.connect(self._accept)
        self.buttons.addWidget(c)
        self.buttons.addWidget(self.go)

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, 'Dataset or bag folder')
        if d:
            self.path.setText(d)

    def _browse_file(self):
        f, _ = QFileDialog.getOpenFileName(self, 'Dataset or bag file', '', BAG_FILES)
        if f:
            self.path.setText(dataset_root(f))

    def values(self) -> dict:
        """The choices; raises ValueError in words when one cannot be used."""
        path = self.path.text().strip()
        if not path:
            raise ValueError('Choose the dataset first (Folder… or File…).')
        if not os.path.exists(path):
            raise ValueError('There is nothing at that path: choose the dataset again.')
        out = {'path': dataset_root(path), 'episodes': parse_episodes(self.episodes.text()),
               'every': int(round(self.every.value())), 'unit': self.unit.value() or 'rad'}
        if self.kind == 'grasps':
            out['mode'] = self.mode.value() or 'closures'
        else:
            out.update(sensor_uid=self.sensor.currentData() or '', camera=self.camera.text().strip() or None,
                       offset_s=self.offset.value() / 1000.0, still_only=not self.moving.isChecked())
        return out

    def _accept(self):
        try:
            self.values()
        except ValueError as e:
            self.hint.setText(str(e))
            return
        self.accept()
