"""Phase 11 integration fixes: each with a regression test (the cable polylines off the UI thread are tested in
test_cables.py, the site's source-only comment lines in test_site.py, the release candidate's publishing in
test_packaging.py)."""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_publication_number_check_takes_a_version_like_token():
    """found() ran float() before the multi-dot test, so '3.17.0' not in the word set raised ValueError."""
    from dexvantage.validation.checks_publication import found
    assert found('3.17.0', [3.17], set(), []) is False
    assert found('3.17.0', [], {'3.17.0'}, []) is True
    assert found('3.2', [3.17], set(), []) is True                # rounded to the digits shown
    assert found('12', [], set(), [12]) is True                   # a list length


def test_a_counted_word_is_singular_for_one():
    from dexvantage.api import _fmt
    assert _fmt(1, 'tokens') == '1 token' and _fmt(2, 'tokens') == '2 tokens' and _fmt(1.5, 'tokens') == '1.5 tokens'
    assert _fmt(1, 's') == '1 s' and _fmt(1, 'ms') == '1 ms' and _fmt(1, 'count') == '1' and _fmt(1, 'm/s') == '1 m/s'


def test_the_version_is_the_release_candidate_everywhere():
    import dexvantage
    ver = dexvantage.__version__
    assert ver == '0.1.0rc1'
    with open(os.path.join(ROOT, 'CITATION.cff'), encoding='utf-8') as f:
        assert f'version: "{ver}"' in f.read()
    with open(os.path.join(ROOT, 'CHANGELOG.md'), encoding='utf-8') as f:
        log = f.read()
    assert re.search(rf'^## {re.escape(ver)} — unreleased$', log, re.M)
    assert 'PLACEHOLDER' not in log
    with open(os.path.join(ROOT, 'README.md'), encoding='utf-8') as f:
        assert ver in f.read()


def test_an_av1_file_that_does_not_decode_is_encoded_again_with_one_thread(tmp_path, monkeypatch):
    """SVT-AV1 now and then writes a stream that does not decode (measured: 2 of 32 and 1 of 40 small datasets
    written four at a time; none of 40 with lp=1). The writer decodes each AV1 file and encodes a bad one again with
    one thread; the frame count of the generation is not doubled by the second pass."""
    from render_support import require
    require()
    from dexvantage.core.datasets import lerobot as L
    from dexvantage.core.robot.samples import sample_project
    from dexvantage.core.sim import episodes as SE
    if L.available_codec('auto')[0] != 'libsvtav1':
        import pytest
        pytest.skip('PyAV here has no libsvtav1 encoder, so the AV1 retry cannot be exercised.')
    real, calls, threads = L._decodes, [], []
    real_init = L._VideoFile.__init__

    def init(self, *a, **k):
        threads.append(a[-1] if len(a) >= 8 else k.get('single_thread', False))
        real_init(self, *a, **k)

    def decodes(path, n):
        calls.append(path)
        return False if len(calls) == 1 else real(path, n)        # the first file "does not decode"
    monkeypatch.setattr(L._VideoFile, '__init__', init)
    monkeypatch.setattr(L, '_decodes', decodes)
    cfg = {'episodes': 1, 'frames': 12, 'max_width': 160, 'seed': 5, 'version': 'v2.1'}
    made = SE.generate(sample_project('so100'), str(tmp_path / 'a'), cfg)
    assert threads[:2] == [False, True] and threads[2:] and not any(threads[2:])
    assert calls[0] == calls[1]                                   # the same file, checked again after the retry
    monkeypatch.setattr(L, '_decodes', real)
    again = SE.generate(sample_project('so100'), str(tmp_path / 'b'), cfg)
    assert made['camera_frames'] == again['camera_frames']
    from dexvantage.core.datasets import open_dataset
    with open_dataset(str(tmp_path / 'a')) as ds:
        ep = ds.load(0)
        for cam in ep.cameras.values():
            assert sum(1 for _ in cam.frames()) == cam.n_frames
