"""Export ▸ Stream to Rerun… (spec §9.4): the scene to an .rrd file or a new Rerun viewer, as a window job
(core/bridges/rerun.py). The viewer is started only here, on the user's choice; the audit result of this session,
when there is one, goes with it as per-episode flags.

    stream(win, mode=None, path=None) -> bool     mode 'save' | 'spawn' (None: ask); True when the job started
"""
from __future__ import annotations

import os

from PyQt6.QtWidgets import QFileDialog, QMessageBox

from ..core.bridges import rerun as RB


def _audit(win):
    try:
        doc = win.ctx.analysis.results.get('audit')
    except AttributeError:
        return None
    return doc if isinstance(doc, dict) and doc.get('episodes') else None


def stream(win, mode: str | None = None, path: str | None = None) -> bool:
    if not RB.available():
        QMessageBox.information(win, 'Stream to Rerun', RB.INSTALL)
        return False
    if mode is None:
        box = QMessageBox(win)
        box.setWindowTitle('Stream to Rerun')
        box.setText('Send the robot, its trajectory, the camera frusta, the min-Z zones and the task volumes to Rerun'
                    + (' with the audit flags of the last dataset audit' if _audit(win) else '') + '.')
        b_save = box.addButton('Save an .rrd file…', QMessageBox.ButtonRole.AcceptRole)
        b_view = box.addButton('Open the Rerun viewer', QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        mode = 'save' if box.clickedButton() is b_save else 'spawn' if box.clickedButton() is b_view else None
        if mode is None:
            return False
    if mode == 'save' and not path:
        path, _ = QFileDialog.getSaveFileName(win, 'Save the scene for Rerun', 'dexvantage.rrd', 'Rerun (*.rrd)')
        if not path:
            return False
        if not path.lower().endswith('.rrd'):
            path += '.rrd'
    snap = win.store.project.snapshot()
    audit = _audit(win)

    def work(progress, cancel):
        progress(0.1, 'the scene')
        return RB.stream(snap, mode, path, audit=audit)

    def done(res):
        win.msg(res['text'] + (' ' + ' '.join(res['notes']) if res['notes'] else ''), 10000)
        win.last_rerun = res

    def failed(message):
        win.msg('Stream to Rerun stopped: ' + message.splitlines()[0], 10000)
    started = win.run_job('rerun', 'Streaming to Rerun' if mode != 'save' else
                          f'Saving {os.path.basename(path)} for Rerun', work, done, failed)
    if not started:
        win.msg('The scene is being sent to Rerun already', 4000)
    return started
