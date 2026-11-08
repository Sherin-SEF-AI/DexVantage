"""Regressions found while integrating Phases 9 and 10 (the samples, the catalogue, the library and research tools
part 1): each test names what went wrong."""
from dexvantage.core import jobs
from dexvantage.core.robot.samples import sample_project

from conftest import pump


def test_a_bracket_for_a_device_without_a_stated_size_is_refused_in_words():
    # the brackets study raised a TypeError in core/brackets.box_of for a sensor with a bracket type whose library
    # device states no size (the Leopard LI-AR0234CS GMSL2 Owl, the ams-OSRAM NanEyeC)
    p = sample_project('ur5e')
    s = p.variant.sensors[1]
    s.model_id, s.kind, s.mount.bracket = 'leopard-li-ar0234cs-gmsl2-owl', 'camera', 'tslot2020'
    jobs.load_studies()
    doc = jobs.run_study('brackets', p)
    row = next(r for r in doc['sensors'] if r['uid'] == s.uid)
    assert row['status'] == 'critical' and 'size' in row['message'] and 'not stated' in row['message']


def test_the_canonical_view_has_named_metrics():
    from dexvantage.core import metrics as M
    M.load_metrics()
    assert {'canonical_view.parallax_p95', 'canonical_view.coverage'} <= set(M.METRICS)
    p = sample_project('ur5e')
    out = M.evaluate(p, ['canonical_view.parallax_p95', 'canonical_view.coverage'])
    assert out['canonical_view.parallax_p95']['value'] > 0 and out['canonical_view.parallax_p95']['unit'] == 'px'
    assert 0 < out['canonical_view.coverage']['value'] <= 100


def test_the_export_dialog_gives_the_report_the_windows_results_and_poses_isaac(app, win):
    from dexvantage.ui.dialogs import export as DX
    win.store.set_project(sample_project('so100'))
    win.ctx.analysis.wait(30)
    pump(app, 0.05)
    have = {sid for sid, d in win.ctx.analysis.results.items() if d.get('revision') == win.store.revision}
    assert have                                       # the live studies ran on this revision
    d = DX.open_dialog(win)
    try:
        d.set_chosen(['report', 'isaac'])
        d.pose.setCurrentIndex(d.pose.findData('screen'))
        opts = d.options()
        assert set(opts['report']['results']) == have and opts['report']['studies'] == 'all'
        assert 'run_missing' not in opts['report'] and 'revision' not in next(iter(opts['report']['results'].values()))
        assert set(opts['isaac']['q']) == {r.uid for r in win.store.project.robots}
        kind, box = d.rows['report'].controls['run_missing']
        box.setChecked(False)
        assert 'studies' not in d.options()['report']             # only the results already computed
    finally:
        d.close()


def test_no_extra_left_unpinned():
    import os
    import re
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    text = open(os.path.join(root, 'pyproject.toml'), encoding='utf-8').read()
    extras = text.split('[project.optional-dependencies]', 1)[1].split('\n[', 1)[0]
    for line in extras.splitlines():
        m = re.match(r'\s*(\w+)\s*=\s*\[(.*)\]', line)
        if m and m.group(1) not in ('test', 'pin', 'lerobot', 'bags'):      # older extras, unpinned on purpose
            assert all('==' in d for d in re.findall(r'"([^"]+)"', m.group(2))), line


def test_open_sample_from_the_menu_builds_it_off_the_ui_thread(app, win):
    # File ▸ Open sample built the project on the UI thread (about 2.5 s for ALOHA 2 the first time)
    import time
    win.store.undo.setClean()
    assert win.open_sample('panda-bimanual', background=True)
    assert 'sample' in win._jobs and win.store.project.name != 'Franka Panda bimanual sample'
    t0 = time.time()
    while 'sample' in win._jobs and time.time() - t0 < 60:
        pump(app, 0.02)
    assert len(win.store.project.robots) == 2 and 'Panda' in win.store.project.name


def test_a_packet_the_threaded_video_decoder_refuses_is_decoded_again_single_threaded(tmp_path, monkeypatch):
    # the full suite saw "Invalid data found when processing input" now and then from the frame-threaded AV1 decoder
    # on files that decode cleanly: the reader tries once more without threads, from the frame it reached
    import numpy as np
    import pytest
    av = pytest.importorskip('av', reason='av (the lerobot extra) is not installed')
    pytest.importorskip('pyarrow', reason='pyarrow (the lerobot extra) is not installed')
    from dexvantage.core import datasets as DS
    from dexvantage.core.datasets import lerobot as L
    from dexvantage.validation import checks_formats as CF
    root = str(tmp_path / 'ds')
    L.write_lerobot(root, CF.lerobot_reference_episodes(np.random.default_rng(7)), 'v2.1', 10, 'reference_arm',
                    CF.LEROBOT_EXTRA)
    ds = DS.open_dataset(root)
    ep = ds.load(ds.episodes()[0].index)
    cam = next(iter(ep.cameras.values()))
    clean = [i for i, _ in cam.frames()]
    real_open, calls = av.open, []

    class Flaky:                                    # a container whose threaded decoder fails after two frames
        def __init__(self, c):
            self.c = c

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return self.c.__exit__(*a)

        def __getattr__(self, k):
            return getattr(self.c, k)

        def decode(self, s):
            calls.append(s.thread_type)
            for k, fr in enumerate(self.c.decode(s)):
                if str(s.thread_type).upper().endswith('AUTO') and k == 2:
                    raise av.error.InvalidDataError(1094995529, 'Invalid data found when processing input')
                yield fr
    monkeypatch.setattr(av, 'open', lambda *a, **k: Flaky(real_open(*a, **k).__enter__()))
    got = [i for i, _ in cam.frames()]
    assert got == clean and len(calls) == 2 and 'NONE' in str(calls[1]).upper()
