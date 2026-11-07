"""'Run on…' for the Designed experiments and Tolerances pages (core/cluster): this machine with N worker processes,
or a Slurm job folder to submit on a cluster; and Gather, which reads a job folder back into the page.

    add_run_on(page, kind)            adds the 'Run on…' button to the page's headline row (kind 'experiment' or
                                      'tolerance'); the page calls nothing else
    run_local(page, jobs)             the page's study in a pool of `jobs` processes (a window job: progress, cancel)
    write_slurm(page, folder, settings, shards)   the job folder (a window job)
    gather(page, folder)              the gathered result shown on the page, or which shards are still to do

All the work runs on a worker thread through the window's job bar (MainWindow.run_job); a result arrives as the
study's own result (AnalysisController.finished), so the page shows it as it shows a run of its own, and its Export
writes it with provenance. Errors are said in a sentence, never as a traceback.

Known limits: the dialog does not submit to Slurm (copy the folder's sbatch command; the batch flag --submit does);
the pool's progress counts items as they finish, not the studies inside them.
"""
from __future__ import annotations

import os
import traceback

from PyQt6.QtWidgets import (QFileDialog, QGridLayout, QHBoxLayout, QLineEdit, QRadioButton, QSpinBox, QVBoxLayout,
                             QWidget)

from ..core import jobs
from ..core.cluster import local as L
from ..core.cluster import slurm as SL
from .dialogs.base import VDialog
from .widgets import button, label

KINDS = ('experiment', 'tolerance')


def _work(page, kind):
    """The page's work, built from a snapshot of the project (on the worker thread)."""
    from ..core import cluster as C
    from ..core import experiments as EX
    p = page.ctx.store.project.snapshot()
    if kind == 'experiment':
        st = EX.settings_of(p)
        if not st.get('factors'):
            raise EX.ExperimentError('The experiment has no factor yet: add one (pick a quantity of the rig).')
        return C.experiment_work(p, st)
    return C.tolerance_work(p)


def _say(page, text, ms=8000):
    page.note.setText(text)
    win = page.ctx.window
    if win is not None:
        win.msg(text, ms)


def _start(page, key, title, fn, on_done):
    win = page.ctx.window
    if win is None:
        return False

    def job(progress, cancel):
        try:
            return {'ok': fn(jobs.Ctx(progress=progress, cancel=cancel))}
        except jobs.StudyError as e:                       # every ClusterError, ExperimentError, SlurmError
            return {'error': str(e)}

    def done(res):
        if 'error' in res:
            _say(page, res['error'])
            return
        on_done(res['ok'])

    def failed(message):
        print(message)
        _say(page, f'{title} stopped with an unexpected error; the details are in the log.')
    return win.run_job(key, title, job, done, failed)


def _show(page, doc):
    doc = dict(doc)
    doc['revision'] = page.ctx.store.revision
    page.ctx.analysis.finished.emit(page.STUDY, doc)


def run_local(page, jobs_n: int, kind: str | None = None) -> bool:
    """Run the page's study on this machine in jobs_n processes; the result is shown when it arrives."""
    kind = kind or page._cluster_kind
    rev = page.ctx.store.revision

    def fn(ctx):
        return L.run_local(_work(page, kind), jobs_n, ctx)

    def done(doc):
        doc = dict(doc)
        doc['revision'] = rev
        page.ctx.analysis.finished.emit(page.STUDY, doc)
        ex = doc.get('execution') or {}
        peak = ex.get('peak_rss_mb_per_worker')
        _say(page, f'{page.TITLE} on {ex.get("workers", jobs_n)} processes in {ex.get("seconds", 0):.1f} s' +
             (f'; peak memory {peak:.0f} MB per worker.' if peak else '.'))
    return _start(page, f'cluster:{page.STUDY}', f'{page.TITLE} on {jobs_n} processes', fn, done)


def write_slurm(page, folder: str, settings: dict | None = None, shards: int | None = None,
                kind: str | None = None) -> bool:
    """Write a Slurm job folder for the page's study."""
    kind = kind or page._cluster_kind

    def fn(ctx):
        out = SL.write_job_folder(_work(page, kind), folder, settings, shards)
        return {k: v for k, v in out.items() if k != 'settings'}

    def done(out):
        page._cluster_last = out
        _say(page, out['text'], 15000)
    return _start(page, f'cluster:{page.STUDY}', 'Writing the Slurm job folder', fn, done)


def gather(page, folder: str) -> bool:
    """Gather a job folder: its result on the page when every shard is done, else which shards are still to do."""
    from ..core.cluster import shards as SH

    def fn(ctx):
        rep = SH.gather(folder, write=True)
        if rep['complete'] and rep['result'].get('study') != page.STUDY:
            raise SH.ClusterError(f'That job folder holds {rep["title"].lower()}, not {page.TITLE.lower()}.')
        return rep

    def done(rep):
        page._cluster_last = SH.report_json(rep)
        if rep['complete']:
            _show(page, rep['result'])
        _say(page, rep['text'], 15000)
    return _start(page, f'cluster:{page.STUDY}', 'Gathering the job folder', fn, done)


class RunOnDialog(VDialog):
    """Where to run: this machine (N worker processes) or a Slurm job folder (its sbatch settings)."""

    def __init__(self, parent, title: str):
        super().__init__(parent, 'Run on…', title, (620, 470), resizable=True)
        lay = QVBoxLayout(self.body)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)
        self.local = QRadioButton('This machine')
        self.slurm = QRadioButton('Write a Slurm job folder (an sbatch array to submit on the cluster)')
        self.local.setChecked(True)
        rec = L.recommended_workers()
        self.workers = QSpinBox()
        self.workers.setRange(1, L.MAX_WORKERS)
        self.workers.setValue(rec)
        self.workers.setToolTip(f'Worker processes (default {rec}: the CPUs, within the memory available at about '
                                f'{L.MEM_PER_WORKER_MB / 1024:.0f} GiB each, a planning default)')
        self.mem_note = label(L.memory_statement(rec), muted=True, wrap=True)
        self.workers.valueChanged.connect(lambda n: self.mem_note.setText(L.memory_statement(int(n))))
        lay.addWidget(self.local)
        w1 = QWidget()
        h1 = QHBoxLayout(w1)
        h1.setContentsMargins(22, 0, 0, 0)
        h1.addWidget(label('Processes', muted=True))
        h1.addWidget(self.workers)
        h1.addStretch(1)
        lay.addWidget(w1)
        lay.addWidget(self.mem_note)
        lay.addWidget(self.slurm)
        w2 = QWidget()
        g = QGridLayout(w2)
        g.setContentsMargins(22, 0, 0, 0)
        g.setHorizontalSpacing(8)
        d = SL.DEFAULTS
        self.folder = QLineEdit()
        self.folder.setPlaceholderText('a new or empty folder on the cluster\'s shared filesystem')
        b = button('Folder…', role='ghost')
        b.clicked.connect(self._browse)
        self.shards = QSpinBox()
        self.shards.setRange(1, SL.MAX_ARRAY_DEFAULT)
        self.shards.setValue(32)
        self.shards.setToolTip('Array tasks (default 32, at most the items; Slurm\'s default MaxArraySize is 1001)')
        self.cpus = QSpinBox()
        self.cpus.setRange(1, 256)
        self.cpus.setValue(d['cpus_per_task'])
        self.mem = QLineEdit(d['mem'])
        self.time = QLineEdit(d['time'])
        self.partition = QLineEdit()
        self.partition.setPlaceholderText('cluster default')
        self.account = QLineEdit()
        self.account.setPlaceholderText('your default')
        self.activate = QLineEdit()
        self.activate.setPlaceholderText('source <your venv>/bin/activate   (or module load …)')
        for wdg, k in ((self.cpus, 'cpus_per_task'), (self.mem, 'mem'), (self.time, 'time'),
                       (self.partition, 'partition'), (self.account, 'account'), (self.activate, 'activate')):
            wdg.setToolTip(SL.SETTINGS[k])
        rows = [('Folder', self.folder, b), ('Shards', self.shards, None), ('CPUs per task', self.cpus, None),
                ('Memory per task', self.mem, None), ('Time limit', self.time, None),
                ('Partition', self.partition, None), ('Account', self.account, None),
                ('Environment', self.activate, None)]
        for r, (t, wdg, extra) in enumerate(rows):
            g.addWidget(label(t, muted=True), r, 0)
            g.addWidget(wdg, r, 1)
            if extra is not None:
                g.addWidget(extra, r, 2)
        lay.addWidget(w2)
        lay.addWidget(label('Defaults are DexVantage\'s, not your site\'s. Submit from the folder with sbatch '
                            'run.sbatch; Gather reads the results back here.', muted=True, wrap=True, small=True))
        lay.addStretch(1)
        self.b_gather = button('Gather a job folder…', role='ghost', tip='Read a job folder\'s shard outputs back')
        self.b_cancel = button('Cancel', role='ghost')
        self.b_ok = button('Run', role='primary')
        self.b_cancel.clicked.connect(self.reject)
        self.b_ok.clicked.connect(self.accept)
        self.b_gather.clicked.connect(self._gather)
        self.choice = None
        self.buttons.addWidget(self.b_gather)
        self.buttons.addWidget(self.b_cancel)
        self.buttons.addWidget(self.b_ok)
        self.local.toggled.connect(self._mode)
        self._mode()

    def _mode(self, *_a):
        on = self.slurm.isChecked()
        for wdg in (self.folder, self.shards, self.cpus, self.mem, self.time, self.partition, self.account,
                    self.activate):
            wdg.setEnabled(on)
        self.workers.setEnabled(not on)
        self.b_ok.setText('Write folder' if on else 'Run')

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, 'Job folder (new or empty)')
        if d:
            self.folder.setText(d)

    def _gather(self):
        d = QFileDialog.getExistingDirectory(self, 'Job folder to gather')
        if d:
            self.choice = {'mode': 'gather', 'folder': d}
            self.accept()

    def slurm_settings(self) -> dict:
        st = {'cpus_per_task': int(self.cpus.value())}
        for k, wdg in (('mem', self.mem), ('time', self.time), ('partition', self.partition),
                       ('account', self.account), ('activate', self.activate)):
            v = wdg.text().strip()
            if v and v != str(SL.DEFAULTS[k]):
                st[k] = v
        if st['cpus_per_task'] == SL.DEFAULTS['cpus_per_task']:
            st.pop('cpus_per_task')
        return st

    def values(self) -> dict:
        if self.choice is not None:
            return self.choice
        if self.slurm.isChecked():
            return {'mode': 'slurm', 'folder': self.folder.text().strip(), 'settings': self.slurm_settings(),
                    'shards': int(self.shards.value())}
        return {'mode': 'local', 'jobs': int(self.workers.value())}


def open_dialog(page):
    """The 'Run on…' button: the dialog, then the job it asks for."""
    try:
        dlg = RunOnDialog(page.ctx.window or page, page.TITLE)
        if not dlg.exec():
            return None
        v = dlg.values()
    except Exception:                                      # noqa: BLE001 — explained, the traceback logged
        traceback.print_exc()
        _say(page, 'The Run on… dialog could not open; the details are in the log.')
        return None
    return apply_choice(page, v)


def apply_choice(page, v: dict):
    if v['mode'] == 'local':
        return run_local(page, v['jobs'])
    if v['mode'] == 'gather':
        return gather(page, v['folder'])
    if not v.get('folder'):
        _say(page, 'Choose a folder for the job (new or empty, on the cluster\'s shared filesystem).')
        return False
    try:
        SL.read_settings(v.get('settings'))
    except SL.SlurmError as e:
        _say(page, str(e))
        return False
    return write_slurm(page, os.path.abspath(v['folder']), v.get('settings'), v.get('shards'))


def add_run_on(page, kind: str):
    """Add 'Run on…' after the page's Run button."""
    assert kind in KINDS, kind
    page._cluster_kind = kind
    page._cluster_last = None
    b = button('Run on…', role='ghost', tip='Run on this machine\'s cores (N processes) or write a Slurm job folder '
                                            'for a cluster; Gather reads a job folder back')
    b.clicked.connect(lambda: open_dialog(page))
    page.head.actions.insertWidget(1, b)
    page.b_run_on = b
    return b
