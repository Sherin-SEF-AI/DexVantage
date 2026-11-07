"""'Run on…' on the Designed experiments and Tolerances pages (ui/cluster_run.py): this machine with N processes, a
Slurm job folder, and Gather back into the page; each runs as a window job, errors are sentences."""
import json
import os
import time

import pytest

CAM = 'sensor:Third-person camera (sample placement)'


def _wait_jobs(app, win, timeout=180.0):
    from conftest import pump
    t0 = time.time()
    while win._jobs and time.time() - t0 < timeout:
        pump(app, 0.05)
    pump(app, 0.05)
    assert not win._jobs, 'the window job did not finish'


def _sample(win):
    from dexvantage.core.robot.samples import sample_project
    p = sample_project('so100')
    p.analysis.studies['experiments'] = {
        'factors': [{'path': CAM + '.pitch', 'lo': -5, 'hi': 5}], 'responses': ['coverage.covered'],
        'design': {'method': 'lhs', 'n': 3, 'seed': 1}, 'analysis': {'bootstrap': 20}}
    p.analysis.studies['tolerance'] = {'n': 3, 'seed': 2, 'metrics': ['coverage.covered']}
    win.store.set_project(p)
    win.ctx.analysis.wait(60)


@pytest.fixture(autouse=True)
def _cold_occlusion_cache():
    from dexvantage.core import occlusion as OC
    OC.CACHE.clear()                     # as a fresh process: tests/test_cluster.py says why
    yield


def test_pages_have_run_on_and_the_dialog_reads_its_choices(app, win):
    from dexvantage.ui import cluster_run as CR
    _sample(win)
    for pid in ('experiments', 'tolerance'):
        win.open_page(pid)
        page = win.pages[pid]
        assert page.b_run_on.text() == 'Run on…'
        assert page.head.actions.indexOf(page.b_run_on) == page.head.actions.indexOf(page.b_run) + 1
    t0 = time.perf_counter()
    dlg = CR.RunOnDialog(win, 'Designed experiments')
    assert (time.perf_counter() - t0) < 0.1                     # cheap enough for the UI thread
    v = dlg.values()
    assert v['mode'] == 'local' and v['jobs'] >= 1 and 'GiB each' in dlg.mem_note.text()
    assert not dlg.mem.isEnabled()
    dlg.slurm.setChecked(True)
    assert dlg.mem.isEnabled() and dlg.b_ok.text() == 'Write folder'
    dlg.folder.setText('/tmp/x')
    dlg.mem.setText('8G')
    dlg.activate.setText('source venv/bin/activate')
    v = dlg.values()
    assert v['mode'] == 'slurm' and v['settings'] == {'mem': '8G', 'activate': 'source venv/bin/activate'}
    dlg.deleteLater()


def test_run_on_this_machine_shows_the_pooled_result(app, win):
    from dexvantage.core import experiments as EX
    from dexvantage.ui import cluster_run as CR
    _sample(win)
    win.open_page('experiments')
    page = win.pages['experiments']
    t0 = time.perf_counter()
    assert CR.apply_choice(page, {'mode': 'local', 'jobs': 2})
    assert time.perf_counter() - t0 < 0.1                       # the work runs on the job thread
    _wait_jobs(app, win)
    assert page.doc is not None and page.doc['execution']['workers'] == 2
    assert 'process pool of 2' in page.doc['execution']['mode'] and 'peak memory' in page.note.text()
    ser = EX.run(win.store.project, EX.settings_of(win.store.project))
    assert [r['responses'] for r in page.doc['runs']] == [r['responses'] for r in ser['runs']]
    assert page.head.state['word'] == ser['headline']['word']
    win.open_page('tolerance')
    tp = win.pages['tolerance']
    assert CR.run_local(tp, 2)
    _wait_jobs(app, win)
    assert tp.doc is not None and tp.doc['done'] == 3 and tp.doc['study'] == 'tolerance'


def test_write_a_slurm_folder_then_gather_it_into_the_page(app, win, tmp_path):
    from dexvantage.core import cluster as C
    from dexvantage.ui import cluster_run as CR
    _sample(win)
    win.open_page('tolerance')
    page = win.pages['tolerance']
    folder = str(tmp_path / 'job')
    assert CR.apply_choice(page, {'mode': 'slurm', 'folder': folder, 'settings': {'mem': '3G'}, 'shards': 2})
    _wait_jobs(app, win)
    assert os.path.exists(os.path.join(folder, 'run.sbatch')) and 'sbatch run.sbatch' in page.note.text()
    assert json.load(open(os.path.join(folder, 'job.json')))['slurm']['mem'] == '3G'
    assert CR.gather(page, folder)                                 # nothing run yet: which shards are missing
    _wait_jobs(app, win)
    assert '2 missing' in page.note.text() and page._cluster_last['missing'] == [0, 1]
    for i in range(2):
        C.run_shard(os.path.join(folder, 'work.json'), i, 2)
    page.doc = None
    assert CR.gather(page, folder)
    _wait_jobs(app, win)
    assert page.doc is not None and page.doc['done'] == 3 and 'Gathered' in page.note.text()
    # a folder of another study is refused in a sentence; a bad setting too, before any job starts
    win.open_page('experiments')
    ep = win.pages['experiments']
    assert CR.gather(ep, folder)
    _wait_jobs(app, win)
    assert 'not designed experiments' in ep.note.text()
    assert CR.apply_choice(ep, {'mode': 'slurm', 'folder': str(tmp_path / 'j2'), 'settings': {'mem': 'lots'}}) \
        is False
    assert 'Slurm size' in ep.note.text()
    assert CR.apply_choice(ep, {'mode': 'slurm', 'folder': '', 'settings': {}}) is False
    assert 'Choose a folder' in ep.note.text()


def test_refusals_are_sentences(app, win, tmp_path):
    from dexvantage.core.project import blank_project
    from dexvantage.ui import cluster_run as CR
    win.store.set_project(blank_project('Empty'))
    win.ctx.analysis.wait(30)
    win.open_page('experiments')
    page = win.pages['experiments']
    assert CR.run_local(page, 2)
    _wait_jobs(app, win)
    assert 'no factor yet' in page.note.text() and 'Traceback' not in page.note.text()
    win.open_page('tolerance')
    tp = win.pages['tolerance']
    assert CR.write_slurm(tp, str(tmp_path / 'j'))
    _wait_jobs(app, win)
    assert 'sensor' in tp.note.text() and 'Traceback' not in tp.note.text()
