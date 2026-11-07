"""Spec-completeness review of Phases 4, 5 and 8: the pieces the review found missing or partial and added
(the batch aliases and audit exports of §8.2/§11, the MJCF scene check of §10, and the others below)."""
import argparse
import json
import os
import subprocess
import sys

import pytest

from dexvantage.core.robot.samples import sample_project

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def _run(*args):
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    return subprocess.run([sys.executable, '-m', 'dexvantage', *args], capture_output=True, text=True, env=env,
                          cwd=ROOT, timeout=180)


# ---------------------------------------------------------------------- §11 flag names of the specification
def test_spec_flag_names_are_aliases():
    from dexvantage import batch
    ap = argparse.ArgumentParser()
    batch.add_arguments(ap)
    a = ap.parse_args(['--teleop-overlap', 'a.json', '--sync', 'b.json'])
    assert a.study__teleop == 'a.json' and a.study__time_sync == 'b.json'
    a = ap.parse_args(['--teleop', 'a.json', '--time-sync', 'b.json'])
    assert a.study__teleop == 'a.json' and a.study__time_sync == 'b.json'


def test_audit_exports_need_the_audit(tmp_path):
    r = _run('--sample', 'so100', '--audit-html', str(tmp_path / 'a.html'))
    assert r.returncode == 2 and 'go with --audit' in r.stderr and 'Traceback' not in r.stderr


def test_audit_html_and_fiftyone_from_batch(tmp_path):
    from dexvantage.core.sim import episodes as SE
    root = str(tmp_path / 'small')
    SE.generate(sample_project('so100'), root, {'episodes': 2, 'frames': 40, 'size': [160, 120], 'seed': 2,
                                                'faults': ['drop']})
    out, html, fo = tmp_path / 'audit.json', tmp_path / 'audit.html', tmp_path / 'fo'
    r = _run('--sample', 'so100', '--audit', str(out), '--dataset', root, '--audit-html', str(html),
             '--fiftyone', str(fo))
    assert r.returncode == 0, r.stdout + r.stderr
    doc = json.loads(out.read_text())
    assert doc['summary']['episodes'] == 2
    page = html.read_text()
    assert page.startswith('<!doctype html>') or page.lower().startswith('<!doctype html')
    assert str(tmp_path) not in page
    samples = json.loads((fo / 'samples.json').read_text())['samples']
    assert len(samples) == 2 * len(doc['cameras']) and all(any(t.startswith('audit:') for t in s['tags'])
                                                           for s in samples)
    # a second export into the same, now non-empty folder is a sentence, exit 2
    r = _run('--sample', 'so100', '--audit', str(out), '--dataset', root, '--fiftyone', str(fo))
    assert r.returncode == 2 and 'not empty' in r.stderr and 'Traceback' not in r.stderr


# ---------------------------------------------------------------------- §10 formats: the MJCF scene loads
def test_mjcf_scene_check_passes(tmp_path):
    import numpy as np

    from dexvantage.validation import checks_formats as CF
    e_p, e_r, det = CF.mjcf_scene_errors(sample_project('ur5e'), str(tmp_path / 'scene'),
                                         np.random.default_rng(0))
    assert e_p < 2e-6 and e_r < 5e-6 and det['intrinsics_px_max'] < 1e-2 and len(det['cameras']) == 6
    assert det['files'] > 0


# ---------------------------------------------------------------------- §8.3 sync against the plan's prediction
def test_recorded_offset_against_the_time_sync_prediction():
    from dexvantage.core.studies import plan_vs_recording as PR
    s = {'sensor_uid': 'u', 'imager': 'RGB', 'name': 'Cam · RGB', 'skew_low_s': 0.004, 'skew_high_s': 0.006,
         'complete': True, 'clock': 'host arrival (default)', 'trigger': 'free'}
    pred = {'streams': [dict(s, imager='Depth'), s]}
    r = PR.sync_prediction(pred, 'u', 'RGB', [{'offset_ms': -5.2, 'sigma_ms': 0.5},
                                              {'offset_ms': -4.9, 'sigma_ms': 0.4}])
    assert r['predicted_offset_ms'] == pytest.approx([-6.0, -4.0]) and r['as_planned'] and r['verdict'] == 'as planned'
    r = PR.sync_prediction(pred, 'u', 'RGB', [{'offset_ms': -50.0, 'sigma_ms': 2.0}])
    assert r['as_planned'] is False and r['verdict'] == 'outside the prediction'
    inc = {'streams': [dict(s, complete=False)]}
    r = PR.sync_prediction(inc, 'u', 'RGB', [{'offset_ms': -50.0, 'sigma_ms': 2.0}])
    assert r['as_planned'] is None and 'lower bound' in r['verdict']
    r = PR.sync_prediction(inc, 'u', 'RGB', [{'offset_ms': 20.0, 'sigma_ms': 2.0}])
    assert r['as_planned'] is None and r['verdict'].startswith('stamps earlier')
    assert PR.sync_prediction(pred, 'other', 'RGB', [])['verdict'] == 'not predicted'
    assert PR.sync_prediction({'why': 'No robot.'}, 'u', 'RGB', [])['why'] == 'No robot.'


# ---------------------------------------------------------------------- §7.8 materials per tag, set on the page
def test_lighting_page_sets_object_materials(app, win):
    from conftest import pump
    win.store.set_project(sample_project('ur5e'))
    pump(app, 0.1)
    page = win.pages['lighting']
    assert page.obj_box.count() == 0 and not page.mat_checks['matte'].isEnabled() and 'No task objects' in \
        page.mat_note.text()
    win.add_object()                                                       # Workspace ▸ Add box object
    pump(app, 0.05)
    objs = win.store.project.workspace.objects
    assert objs and page.obj_box.count() == len(objs)
    o = objs[0]
    page.obj_box.setCurrentIndex(page.obj_box.findData(o.uid))
    before = list(o.materials)
    n0 = win.store.undo.count()
    page.mat_checks['glossy'].setChecked('glossy' not in before)          # a click: one undo step
    after = list(win.store.project.workspace.objects[0].materials)
    assert win.store.undo.count() == n0 + 1 and ('glossy' in after) != ('glossy' in before)
    win.store.undo.undo()
    assert list(win.store.project.workspace.objects[0].materials) == before
    assert page.mat_checks['glossy'].isChecked() == ('glossy' in before)
    # removing the last tag leaves matte, never no material
    for t in list(win.store.project.workspace.objects[0].materials):
        page.set_material(t, False, o.uid)
    assert list(win.store.project.workspace.objects[0].materials) == ['matte']


# ---------------------------------------------------------------------- §8.2 LeRobot from the Hub, streamed
class _FakeHub:
    """Serves a local LeRobot folder the way the Hub's HTTP interface does (core/datasets/hub.py), and records
    every URL asked for. No network."""

    def __init__(self, root, repo='org/so100-synth', sha='0123abcd', page=2, gated=False, serve_ranges=True):
        self.root, self.repo, self.sha, self.page, self.gated = root, repo, sha, page, gated
        self.serve_ranges = serve_ranges
        self.urls = []

    def __call__(self, url, headers=None):
        import io
        import urllib.error
        import urllib.parse
        self.urls.append(url)
        self.ranges = getattr(self, 'ranges', [])
        ep = 'https://huggingface.co'

        class R(io.BytesIO):
            def __init__(self, data, headers=None, status=200):
                super().__init__(data)
                self.headers = headers or {}
                self.status = status

        def nf():
            return urllib.error.HTTPError(url, 404, 'Not Found', {}, None)
        api = f'{ep}/api/datasets/{self.repo}/'
        if url.startswith(api + 'revision/'):
            if url.endswith('/nope'):
                raise nf()
            return R(json.dumps({'sha': self.sha, 'private': False, 'gated': 'auto' if self.gated else False,
                                 'cardData': {'license': 'apache-2.0'}}).encode())
        if url.startswith(api + 'tree/'):
            u = urllib.parse.urlsplit(url)
            prefix = urllib.parse.unquote(u.path.split('/tree/', 1)[1].split('/', 1)[1])
            q = urllib.parse.parse_qs(u.query)
            files = sorted(os.path.relpath(os.path.join(dp, f), self.root).replace(os.sep, '/')
                           for dp, _, fs in os.walk(os.path.join(self.root, prefix)) for f in fs)
            k = int(q.get('cursor', ['0'])[0])
            chunk = files[k:k + self.page]
            items = [{'type': 'file', 'path': f, 'size': 1} for f in chunk] + [{'type': 'directory', 'path': prefix}]
            hdr = {}
            if k + self.page < len(files):
                hdr['Link'] = f'<{u.scheme}://{u.netloc}{u.path}?recursive=true&cursor={k + self.page}>; rel="next"'
            return R(json.dumps(items).encode(), hdr)
        res = f'{ep}/datasets/{self.repo}/resolve/{self.sha}/'
        if url.startswith(res):
            rel = urllib.parse.unquote(url[len(res):])
            p = os.path.join(self.root, rel)
            if not os.path.isfile(p):
                raise nf()
            with open(p, 'rb') as fh:
                data = fh.read()
            rg = (headers or {}).get('Range')
            if rg and self.serve_ranges:
                lo, hi = (int(x) for x in rg.split('=')[1].split('-'))
                self.ranges.append((rel, lo, hi))
                return R(data[lo:hi + 1], {'Content-Range': f'bytes {lo}-{hi}/{len(data)}'}, 206)
            return R(data)
        raise nf()


@pytest.fixture(scope='module')
def synth_v30(tmp_path_factory):
    from dexvantage.core.sim import episodes as SE
    root = str(tmp_path_factory.mktemp('hub') / 'src')
    SE.generate(sample_project('so100'), root, {'episodes': 3, 'frames': 30, 'size': [96, 72], 'seed': 4,
                                                'faults': [], 'version': 'v3.0'})
    return root


def test_hub_refs():
    from dexvantage.core.datasets import hub as H
    assert H.parse_hub_ref('hf://datasets/lerobot/svla_so100_pickplace') == ('lerobot/svla_so100_pickplace', 'main')
    assert H.parse_hub_ref('hf://datasets/org/name@v2.1') == ('org/name', 'v2.1')
    assert H.parse_hub_ref('https://huggingface.co/datasets/org/name/tree/abc123') == ('org/name', 'abc123')
    assert H.parse_hub_ref('https://huggingface.co/datasets/org/name') == ('org/name', 'main')
    for bad in ('org/name', '/data/org/name', 'hf://models/org/name', 'hf://datasets/name', None):
        assert H.parse_hub_ref(bad) is None
    assert H.hub_ref('org/name') == 'hf://datasets/org/name' and H.hub_ref('org/name', 'r1').endswith('@r1')


def test_hub_dataset_streams_only_what_it_reads(synth_v30, tmp_path):
    import numpy as np

    from dexvantage.core.datasets import lerobot as L
    from dexvantage.core.datasets import open_dataset
    fake = _FakeHub(synth_v30)
    cache = str(tmp_path / 'cache')
    ds = L.LeRobotDataset('hf://datasets/org/so100-synth', opener=fake, cache=cache)
    assert ds.version == 'v3.0' and len(ds.episodes()) == 3
    assert ds.source == {'kind': 'hub', 'repo_id': 'org/so100-synth', 'revision': 'main', 'commit': '0123abcd',
                         'license': 'apache-2.0'}
    fetched = [u for u in fake.urls if '/resolve/' in u]
    assert not any('/data/' in u or '/videos/' in u for u in fetched)          # nothing but metadata on open
    fake.page = 1                                                                # a paged file list
    vids = ds.hub.list('videos')
    assert any('cursor=' in u for u in fake.urls) and len(vids) >= 2
    assert vids == sorted(os.path.relpath(os.path.join(dp, f), synth_v30).replace(os.sep, '/')
                          for dp, _, fs in os.walk(os.path.join(synth_v30, 'videos')) for f in fs)
    e = ds.load(1)
    with open_dataset(synth_v30) as local:
        e_local = local.load(1)
    assert np.array_equal(e.q, e_local.q) and np.array_equal(e.t, e_local.t)
    for k, cs in e.cameras.items():
        assert np.array_equal(cs.timestamps, e_local.cameras[k].timestamps)
        a = dict(cs.frames([0, len(cs.timestamps) - 1]))
        b = dict(e_local.cameras[k].frames([0, len(cs.timestamps) - 1]))
        assert all(np.array_equal(a[i], b[i]) for i in a)
    assert any('/data/' in u for u in fake.urls) and any('/videos/' in u for u in fake.urls)
    vid_ranges = [r for r in fake.ranges if r[0].startswith('videos/')]
    assert vid_ranges and not os.path.isdir(os.path.join(ds.hub.root, 'videos'))   # read in place, not saved
    assert ds.hub.bytes_ranged > 0
    # a server that ignores range requests: the whole video file is fetched instead, with the same frames
    whole = _FakeHub(synth_v30, serve_ranges=False)
    ds3 = L.LeRobotDataset('hf://datasets/org/so100-synth', opener=whole, cache=str(tmp_path / 'cache3'))
    e3 = ds3.load(1)
    k = next(iter(e3.cameras))
    assert os.path.isdir(os.path.join(ds3.hub.root, 'videos'))
    assert np.array_equal(dict(e3.cameras[k].frames([0]))[0], dict(e_local.cameras[k].frames([0]))[0])
    n = len(fake.urls)
    ds2 = L.LeRobotDataset('hf://datasets/org/so100-synth', opener=fake, cache=cache)
    ds2.load(1)
    assert all('/data/' not in u for u in fake.urls[n:])                         # the data file is cached
    assert ds2.truth() is not None and ds2.truth()['schema'] == L.TRUTH_SCHEMA


def test_hub_refusals_are_sentences(synth_v30, tmp_path):
    from dexvantage.core.datasets import DatasetError
    from dexvantage.core.datasets import lerobot as L
    with pytest.raises(DatasetError, match='gated') as e:
        L.LeRobotDataset('hf://datasets/org/so100-synth', opener=_FakeHub(synth_v30, gated=True),
                         cache=str(tmp_path))
    assert 'token' in str(e.value)
    with pytest.raises(DatasetError, match='check the dataset'):
        L.LeRobotDataset('hf://datasets/org/so100-synth@nope', opener=_FakeHub(synth_v30), cache=str(tmp_path))

    def offline(url):
        import urllib.error
        raise urllib.error.URLError(OSError('Network is unreachable'))
    with pytest.raises(DatasetError, match='could not be reached'):
        L.LeRobotDataset('hf://datasets/org/so100-synth', opener=offline, cache=str(tmp_path))


def test_audit_reads_a_hub_dataset(synth_v30, tmp_path, monkeypatch):
    from dexvantage.core import jobs
    from dexvantage.core.datasets import hub as H
    from dexvantage.core.studies import audit as AU
    fake = _FakeHub(synth_v30)
    monkeypatch.setattr(H, '_default_opener', fake)
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'xdg'))
    p = sample_project('so100')
    AU.use_dataset(p, 'hf://datasets/org/so100-synth')
    assert AU.dataset_path(p) == 'hf://datasets/org/so100-synth'
    assert AU.fingerprint('hf://datasets/org/so100-synth') != AU.fingerprint('hf://datasets/org/so100-synth@r2')
    doc = jobs.run_study('audit', p)
    assert doc['summary']['episodes'] == 3
    assert doc['dataset']['source']['commit'] == '0123abcd' and doc['dataset']['name'] == 'so100-synth'
    assert os.path.isdir(os.path.join(str(tmp_path / 'xdg'), 'DexVantage', 'hub', 'org--so100-synth', '0123abcd'))


def test_audit_page_adds_a_hub_dataset(app, win, synth_v30, tmp_path, monkeypatch):
    from conftest import pump
    from dexvantage.core.datasets import hub as H
    monkeypatch.setattr(H, '_default_opener', _FakeHub(synth_v30))
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'xdg'))
    win.store.set_project(sample_project('so100'))
    page = win.pages['audit']
    assert page.add_hub_dataset('not a name') is None
    assert page.add_hub_dataset('org/so100-synth') == 'hf://datasets/org/so100-synth'
    for _ in range(200):
        pump(app, 0.05)
        if (win.store.project.data or {}).get('datasets'):
            break
    d = win.store.project.data['datasets'][-1]
    assert d['path'] == 'hf://datasets/org/so100-synth' and d['name'] == 'org/so100-synth'
    assert d['format'] == 'lerobot' and d['version'] == 'v3.0' and d['episodes'] == 3


def test_remote_file_blocks_match_the_bytes(synth_v30, tmp_path, monkeypatch):
    import numpy as np

    from dexvantage.core.datasets import hub as H
    monkeypatch.setattr(H, 'BLOCK', 1000)
    monkeypatch.setattr(H, 'MAX_BLOCKS', 2)
    fake = _FakeHub(synth_v30)
    m = H.HubMirror('org/so100-synth', opener=fake, cache=str(tmp_path))
    rel = next(r for r in m.list('videos') if r.endswith('.mp4'))
    data = open(os.path.join(synth_v30, rel), 'rb').read()
    f = m.remote(rel)
    assert f is not None and f.size == len(data) > 3000
    rng = np.random.default_rng(1)
    for _ in range(40):
        a = int(rng.integers(0, len(data)))
        n = int(rng.integers(1, 3500))
        f.seek(a)
        assert f.read(n) == data[a:a + n]
    f.seek(-10, 2)
    assert f.read() == data[-10:] and f.read(5) == b''
    assert len(m._blocks) <= 2
    assert H.HubMirror('org/so100-synth', opener=_FakeHub(synth_v30, serve_ranges=False),
                       cache=str(tmp_path)).remote(rel) is None


# ---------------------------------------------------------------------- §9.1 the lighting page's lights in the scene
def test_scene_lights_follow_the_lighting_contract():
    import numpy as np

    from dexvantage.core import lights as LT
    from dexvantage.core.sim.scene import build_scene
    p = sample_project('ur5e')
    down = [1.0, 0.0, 0.0, 0.0]                         # the lighting page's "faces down" (local +z to world −z)
    panel = LT.new_light('panel', [0.4, 0.0, 1.5], 'Panel', quaternion=down)
    spot = dict(LT.new_light('spot', [0.0, 0.5, 1.2], 'Spot', quaternion=down),
                beam_deg={'value': 40.0, 'source': 'test'}, dim=0.5)
    off = dict(LT.new_light('point', [0.0, 0.0, 2.0], 'Off'), enabled=False)
    p.workspace.lights = [LT.clean_light(x) for x in (panel, spot, off)]
    b = build_scene(p)
    m = b.model
    names = [m.light(i).name for i in range(m.nlight)]
    assert len(names) == 2 and not any('Off' in n for n in names)
    assert any('is off' in n for n in b.notes)
    for i in range(m.nlight):
        assert m.light_dir[i][2] == pytest.approx(-1.0, abs=1e-9)          # shining down, as the study lights
    i_panel = next(i for i, n in enumerate(names) if panel['uid'] in n)
    i_spot = next(i for i, n in enumerate(names) if spot['uid'] in n)
    assert m.light_cutoff[i_panel] == pytest.approx(90.0) and m.light_exponent[i_panel] == pytest.approx(1.0)
    assert m.light_cutoff[i_spot] == pytest.approx(20.0) and m.light_exponent[i_spot] == pytest.approx(0.0)
    assert np.allclose(m.light_diffuse[i_spot], 0.35) and np.allclose(m.light_diffuse[i_panel], 0.7)
