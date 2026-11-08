"""The Splat views path as a new user meets it (docs/guide/splat.md): the fit, replay and augmentation can be stopped
from the page and the status bar; a cancelled run says so (not "could not run") and keeps the result shown before;
the Fit from choice survives choosing the dataset; the device can be chosen on the page; a new proposed pose reuses the
fitted rig (its cache key leaves out the settings that do not change the fit); the voxel thinning that seeds a fit
gives the same result as np.unique over rows without holding the GIL for long; and the CUDA probe runs in a child
process. No test here fits a rig or needs PyTorch."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from conftest import pump

from dexvantage.core.sim import splat_views as SV
from dexvantage.core.splat_reproject import voxel_centroids


def _rows_reference(P, size, values=None):
    """voxel_centroids written out again the slow way: np.unique over the key rows and np.add.at."""
    P = np.asarray(P, float).reshape(-1, 3)
    size = np.broadcast_to(np.asarray(size, float), (len(P),))
    key = np.floor(P / size[:, None]).astype(np.int64)
    key = np.concatenate([key, np.round(np.log2(size) * 4).astype(np.int64)[:, None]], 1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    out = np.zeros((len(cnt), 3))
    np.add.at(out, inv, P)
    mv = None
    if values is not None:
        V = np.asarray(values, float).reshape(len(P), -1)
        mv = np.zeros((len(cnt), V.shape[1]))
        np.add.at(mv, inv, V)
        mv /= cnt[:, None]
    return out / cnt[:, None], cnt, mv, inv


@pytest.mark.parametrize('sizes', ['one', 'mixed'])
def test_voxel_centroids_match_the_row_unique_reference(sizes):
    rng = np.random.default_rng(3)
    P = rng.normal(size=(20000, 3)) * [0.3, 0.2, 0.1] + [0.4, -0.1, 0.2]
    V = rng.random((20000, 3))
    size = 0.01 if sizes == 'one' else rng.choice([0.005, 0.01, 0.04], len(P))
    got, want = voxel_centroids(P, size, V), _rows_reference(P, size, V)
    for a, b in zip(got, want):
        assert a.shape == b.shape and np.allclose(a, b, rtol=0, atol=1e-12)
    assert np.array_equal(got[1], want[1]) and np.array_equal(got[3], want[3])
    c, cnt, mv, inv = voxel_centroids(P, 0.02)                    # without values
    assert mv is None and cnt.sum() == len(P) and len(inv) == len(P)


def test_a_new_proposed_pose_reuses_the_fitted_rig():
    import dexvantage as dv
    from dexvantage.api_splat import _project
    rig = dv.sample('so100')

    def key(**given):
        p = _project(rig, None, dict(source='demo', **given))
        return SV._rig_key(p, SV.settings_of(p), 'demo', None)
    base = key()
    assert key(target_offset_m=[-0.03, 0.0, 0.02], target_rpy_deg=[0.0, 8.0, 0.0]) == base
    assert key(lpips='off', augment_n=8, replay_scale=0.5, psnr_good_db=30.0) == base
    assert key(iterations=10) != base and key(seed=1) != base and key(device='cpu') != base


def test_fit_from_and_device_survive_choosing_the_dataset(app, win, tmp_path):
    assert win.open_sample('so100')
    page = win.pages['splat']
    page.src_box.setCurrentIndex(page.src_box.findData('demo'))
    page.dev_box.setCurrentIndex(page.dev_box.findData('cpu'))
    page.choose_dataset(str(tmp_path))
    pump(app, 0.02)
    assert page.src_box.currentData() == 'demo' and page.dev_box.currentData() == 'cpu'
    f = page.form()
    assert f['source'] == 'demo' and f['device'] == 'cpu'
    # the project's own stored source still shows when it changes (an API call, an undo to another project)
    win.store.edit('analysis', lambda p: p.analysis.studies['splat_views'].update(source='episode'), 'test')
    pump(app, 0.02)
    assert page.src_box.currentData() == 'episode'


def test_without_cuda_the_page_points_at_the_device_choice(app, win, monkeypatch):
    from dexvantage.ui.pages import splat as SP
    monkeypatch.setattr(SV, 'requirement', lambda: None)
    monkeypatch.setitem(SP._CUDA, 'ok', False)
    assert win.open_sample('so100')
    page = win.pages['splat']
    page._sync()
    assert 'set Device to Processor' in page.avail.text() and page.avail.isVisibleTo(page)


def test_the_cuda_probe_runs_in_a_child_process(monkeypatch):
    import subprocess

    from dexvantage.ui.pages import splat as SP
    seen = []

    class R:
        returncode = 3
    monkeypatch.setattr(subprocess, 'run', lambda args, **kw: seen.append(args) or R())
    monkeypatch.setattr(SP, '_CUDA', {})

    class Sig:
        class cuda:
            emit = staticmethod(lambda: seen.append('emitted'))
    SP._probe_cuda(Sig)
    t0 = time.time()
    while 'emitted' not in seen and time.time() - t0 < 10:
        time.sleep(0.01)
    assert seen[0][1:] == ['-m', 'dexvantage.core.cuda_probe'] and seen[-1] == 'emitted' and SP._CUDA['ok'] is False


def test_the_cuda_probe_module_answers_with_its_exit_code():
    import subprocess
    import sys
    r = subprocess.run([sys.executable, '-m', 'dexvantage.core.cuda_probe'], capture_output=True, timeout=300)
    assert r.returncode in (0, 3) and not r.stderr


def test_a_running_fit_can_be_cancelled_from_the_page(app, win, monkeypatch):
    assert win.open_sample('so100')
    page = win.pages['splat']
    an = win.ctx.analysis
    asked = []
    monkeypatch.setattr(an, 'cancel', lambda sid=None: asked.append(sid))
    an.started.emit('splat_views')
    pump(app, 0.02)
    assert page.b_cancel.isVisibleTo(page) and not page.b_run.isEnabled() and not page.b_replay.isEnabled()
    page.cancel()
    assert asked == ['splat_views']
    an.failed.emit('splat_views', 'StudyError: The splat fit was cancelled; nothing was kept.\nTraceback ...')
    pump(app, 0.02)
    assert page.head.status.text.text() == 'cancelled' and 'Run starts the fit again' in page.head.explain.text()
    # Run is enabled again unless the splat extra is missing here (CI installs no PyTorch: Run then stays off and
    # the headline says why)
    assert not page.b_cancel.isVisibleTo(page) and page.b_run.isEnabled() == (page._avail() is None)
    assert 'Running' not in page.note.text()


def test_a_cancelled_rerun_keeps_the_result_shown_and_a_failure_clears_it(app, win):
    assert win.open_sample('so100')
    page = win.pages['splat']
    doc = {'headline': {'value': 23.1, 'status': 'warning', 'word': 'visible differences (synthetic capture)'},
           'quality': {'held_out_views': {'psnr_db': 23.1, 'ssim': 0.936, 'views': 6}}, 'settings': {}}
    page._finished('splat_views', doc)
    pump(app, 0.02)
    an = win.ctx.analysis
    an.failed.emit('splat_views', 'StudyError: The splat fit was cancelled; nothing was kept.')
    pump(app, 0.02)
    assert page.head.figure.text() == '23.1' and page.qual.topLevelItemCount() == 1
    assert "previous run's" in page.note.text()
    an.failed.emit('splat_views', 'StudyError: There is no CUDA device here, and the splat fit runs on a CUDA GPU.')
    pump(app, 0.02)
    assert page.head.status.text.text() == 'could not run' and page.qual.topLevelItemCount() == 0
    assert page.doc is None and not page.b_apply.isEnabled()


def test_a_replay_can_be_cancelled_from_the_page_and_the_status_bar(app, win, monkeypatch, tmp_path):
    assert win.open_sample('so100')
    page = win.pages['splat']
    monkeypatch.setattr(SV, 'requirement', lambda: None)
    started = threading.Event()

    def fake_replay(project, out_dir, settings=None, *, progress=None, cancel=None):
        started.set()
        for k in range(1000):
            if cancel():
                raise SV.StudyError('The replay was cancelled; nothing was kept.')
            progress(k / 1000, f'Rendering frame {k + 1} of 1000')
            time.sleep(0.005)
        return {'frames': 1000, 'video': {'codec': 'h264', 'file': 'x.mp4'}, 'root': str(tmp_path)}
    monkeypatch.setattr(SV, 'replay', fake_replay)
    for stop in (page.cancel, win.cancel_jobs):                   # the page's Cancel, then the status bar's (Ctrl+.)
        job = page.write('replay', str(tmp_path))
        assert job is not None and started.wait(10)
        pump(app, 0.05)
        assert page.b_cancel.isVisibleTo(page) and not page.b_run.isEnabled()
        assert win.job_label.text().startswith('Writing the splat replay') and win.job_cancel.isVisibleTo(win)
        stop()
        job.join(10)
        pump(app, 0.1)
        assert not job.is_alive() and page.note.text() == 'The replay was cancelled; nothing was kept.'
        assert not page.b_cancel.isVisibleTo(page) and page.b_replay.isEnabled()
        started.clear()


def test_the_status_bar_leaves_analysing_after_a_failed_run(app, win):
    an = win.ctx.analysis
    an._busy['splat_views'] = threading.Event()
    an.started.emit('splat_views')
    pump(app, 0.02)
    assert win.state_label.text() == 'analysing…'
    an._busy.pop('splat_views')
    an.failed.emit('splat_views', 'StudyError: The splat fit was cancelled; nothing was kept.')
    pump(app, 0.02)
    assert win.state_label.text() != 'analysing…'


def test_a_pose_entered_after_the_run_says_run_renders_it(app, win):
    assert win.open_sample('so100')
    page = win.pages['splat']
    uid = page.cam_box.itemData(0)
    page.set_proposal((-0.03, 0.0, 0.02), (0.0, 8.0, 0.0), sensor=uid)
    page.doc = {'headline': {}, 'settings': {'target_sensor': uid, 'target_offset_m': [-0.03, 0.0, 0.02],
                                             'target_rpy_deg': [0.0, 8.0, 0.0]}}
    page._apply_state()
    assert not page.stale_label.isVisibleTo(page)                   # the images are for the pose entered
    page.set_proposal((-0.05, 0.0, 0.02), (0.0, 8.0, 0.0), sensor=uid)
    assert page.stale_label.isVisibleTo(page) and 'Run to render' in page.stale_label.text()
