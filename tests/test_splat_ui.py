"""The Splat views page (ui/pages/splat.py): registered with its study, the headline figure's tooltip naming the
model and its inputs, the before/after images with a 'synthetic' badge on every splat render, the quality table, the
replay written from the page in the background, and the page without the splat extra (it says so; its actions are
off). The result comes from a CPU fit of seconds on the SO-100 sample (MuJoCo renders: tests marked mujoco_render
skip with the reason where MuJoCo cannot render offscreen)."""
from __future__ import annotations

import os
import time

import numpy as np
import pytest

from conftest import pump

from dexvantage.core import jobs
from dexvantage.core.sim import splat_views as SV

TINY = {'device': 'cpu', 'iterations': 10, 'max_gaussians': 500, 'demo_views': 12, 'demo_width': 40,
        'demo_height': 30, 'demo_q_states': 2, 'holdout_q': 2, 'preview_frames': 2, 'thumb_width': 64,
        'views_per_camera': 10, 'width': 48, 'min_views': 6, 'lpips': 'off'}


def test_the_page_is_registered_and_says_what_it_needs(app, win):
    from dexvantage.ui.pages import PAGES
    assert 'splat' in PAGES and PAGES['splat'].cls.STUDY == 'splat_views' and PAGES['splat'].menu == 'Simulate'
    assert win.open_sample('so100')
    page = win.pages['splat']
    win.open_page('splat')
    pump(app, 0.05)
    tip = page.head.figure.toolTip()
    assert 'Model: Splat views (core/studies/splat_views.py)' in tip and 'Inputs: ' in tip
    assert page.cam_box.count() == 2 and 'on a link' in page.cam_box.itemText(0)
    page.set_proposal((0.0, 0.0, 0.03), (0.0, 5.0, 0.0))
    f = page.form()
    assert f['target_offset_m'] == [0.0, 0.0, 0.03] and f['target_rpy_deg'] == [0.0, 5.0, 0.0]
    assert f['target_sensor'] == page.cam_box.itemData(0) and f['source'] == 'auto'
    assert 'none chosen' in page.dataset_label.text()


def test_without_the_splat_extra_the_page_says_so(app, win, monkeypatch):
    monkeypatch.setattr(SV, 'requirement', lambda: 'Splat views need PyTorch: the splat extra is not installed.')
    assert win.open_sample('so100')
    page = win.pages['splat']
    page._sync()
    page.empty()
    assert page.head.status.text.text() == 'splat extra not installed'
    assert not page.b_run.isEnabled() and not page.b_replay.isEnabled() and not page.b_augment.isEnabled()
    assert 'splat extra is not installed' in page.avail.text()
    page.run()
    assert page.head.status.text.text() == 'splat extra not installed'


def test_badge_marks_splat_images():
    from dexvantage.ui.pages.splat import badge_pixmap
    img = np.full((30, 40, 3), 255, np.uint8)
    a = badge_pixmap(img, True, 120).toImage()
    b = badge_pixmap(img, False, 120).toImage()
    assert a.width() == 120 and a.pixelColor(8, 10) != b.pixelColor(8, 10)        # the badge's dark box
    assert b.pixelColor(8, 10).red() == 255


@pytest.mark.mujoco_render
def test_a_result_fills_the_page_and_replay_is_written(app, win, tmp_path):
    pytest.importorskip('torch', reason='PyTorch is not installed (the splat extra): the page\'s result is not '
                                        'tested')
    assert win.open_sample('so100')
    page = win.pages['splat']
    win.open_page('splat')
    snap = win.store.project.snapshot()
    snap.analysis.studies['splat_views'] = dict(TINY, source='demo', target_offset_m=[0.0, 0.0, 0.04])
    doc = jobs.run_study('splat_views', snap)
    page._finished('splat_views', doc)
    pump(app, 0.05)
    assert page.head.figure.text() == f'{doc["headline"]["value"]:.1f}' and 'SSIM' in page.head.unit.text()
    assert page.head.status.text.text() == doc['headline']['word'] and 'synthetic' in page.head.status.text.text()
    tip = page.head.figure.toolTip()
    assert tip.startswith('PSNR of the splat rig') and 'Model: Splat views' in tip and 'iterations 10' in tip
    assert page.qual.topLevelItemCount() >= 3 and page.qual.topLevelItem(0).text(0) == 'Held-out views'
    assert page.frame_box.count() == 2
    for k in ('current', 'proposed'):
        assert not page.pics[k].pixmap().isNull()
    assert page.pics['recorded'].text() == 'not recorded'
    ctx = page.page_context()
    assert ctx['synthetic'] and ctx['quality']['held_out_views']['psnr_db'] == doc['headline']['value']
    # replay without a dataset: refused in a sentence, in the background, never a traceback
    t = page.write('replay', str(tmp_path))
    t.join(120)
    pump(app, 0.1)
    assert 'Replay needs a recorded LeRobot episode' in page.note.text()
    assert page.b_replay.isEnabled()


@pytest.mark.mujoco_render
def test_replay_from_the_page_writes_the_synthetic_folder(app, win, tmp_path):
    pytest.importorskip('torch', reason='PyTorch is not installed (the splat extra): the replay is not tested')
    from dexvantage.core.sim import episodes as EPS
    assert win.open_sample('so100')
    root = str(tmp_path / 'eps')
    EPS.generate(win.store.project, root, {'episodes': 1, 'frames': 20, 'max_width': 96, 'faults': [],
                                           'codec': 'h264'})
    page = win.pages['splat']
    page.choose_dataset(root)
    assert win.store.project.analysis.studies['splat_views']['dataset'] == root
    page.ctx.store.edit('analysis', lambda p: p.analysis.studies['splat_views'].update(
        dict(TINY, replay_max_frames=6, replay_scale=0.25, codec='h264')), 'test settings')
    t = page.write('replay', str(tmp_path / 'out'))
    t0 = time.time()
    while t.is_alive() and time.time() - t0 < 180:
        pump(app, 0.05)
    pump(app, 0.1)
    assert page.note.text().startswith('Replay written (6 frames'), page.note.text()
    assert os.path.isfile(os.path.join(page.last_written['root'], page.last_written['video']['file']))


def test_apply_moves_the_mount_as_proposed(app, win):
    assert win.open_sample('so100')
    page = win.pages['splat']
    page.doc = {'headline': {}}                                   # a result is shown (its figures are not needed)
    uid = page.cam_box.itemData(0)
    before = win.store.project.variant.sensor(uid).mount.pose.matrix()
    page._apply_state()
    assert not page.b_apply.isEnabled()                           # no change entered
    page.set_proposal((0.0, 0.0, 0.03), (0.0, 0.0, 10.0))
    assert page.b_apply.isEnabled()
    n0 = win.store.undo.count()
    page.preview()
    pose = page.apply()
    assert pose is not None and win.store.undo.count() == n0 + 1
    after = win.store.project.variant.sensor(uid).mount.pose.matrix()
    d = np.linalg.inv(before) @ after
    assert np.allclose(d[:3, 3], [0, 0, 0.03], atol=1e-9) and np.degrees(np.arctan2(d[1, 0], d[0, 0])) == \
        pytest.approx(10.0, abs=1e-6)
    assert page.form()['target_offset_m'] == [0.0, 0.0, 0.0] and not page.b_apply.isEnabled()
    win.store.undo.undo()
    assert np.allclose(win.store.project.variant.sensor(uid).mount.pose.matrix(), before)
