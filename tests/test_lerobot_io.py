"""LeRobot v2.1 and v3.0 (core/datasets/lerobot.py): the writer's layout against the format as the LeRobot source
defines it, round trips of every column, the timestamps and the frame pixels, packed v3.0 files, Hub snapshot folders,
the version refusal and the plain errors."""
import builtins
import json
import os
import shutil

import numpy as np
import pytest

pytest.importorskip('pyarrow', reason='pyarrow (the lerobot extra) is not installed')
pytest.importorskip('av', reason='av (the lerobot extra) is not installed')

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from dexvantage.core import datasets as DS  # noqa: E402
from dexvantage.core.datasets import episode as E  # noqa: E402
from dexvantage.core.datasets import lerobot as L  # noqa: E402
from dexvantage.validation import checks_formats as CF  # noqa: E402

CAM = CF.LEROBOT_CAM
VERSIONS = ('v2.1', 'v3.0')


@pytest.fixture
def src():
    return CF.lerobot_reference_episodes(np.random.default_rng(7))


def _write(tmp_path, src, version, name='ds', **kw):
    root = str(tmp_path / f'{name}_{version}')
    s = L.write_lerobot(root, src, version, 10, 'reference_arm', CF.LEROBOT_EXTRA, **kw)
    return root, s


def _psnr(a, b):
    return CF._psnr(a, b)


def _episode(n=6, fps=10, size=(48, 64), key=CAM, ts=None, task='t'):
    t = np.arange(n) / fps
    ts = t.copy() if ts is None else np.asarray(ts, float)
    frames = [CF.lerobot_pattern(i, *size) for i in range(len(ts))]

    def dec(idx):
        for i in (range(len(frames)) if idx is None else idx):
            yield i, frames[i]
    cam = E.CameraStream(key, size[1], size[0], float(fps), ts, _decode=dec)
    return E.Episode('e', 0, t, np.zeros((n, 2)), ['a', 'b'], fps=float(fps), task=task, cameras={key: cam})


# --- round trips --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('version', VERSIONS)
def test_round_trip_every_column_timestamps_and_pixels(tmp_path, src, version):
    root, summary = _write(tmp_path, src, version)
    assert summary['codec'] in ('AV1 (libsvtav1)', 'H.264 (libx264)') and summary['frames'] == 29
    ds = DS.open_dataset(root)
    assert isinstance(ds, L.LeRobotDataset) and ds.format == 'lerobot' and ds.version == version
    refs = ds.episodes()
    assert [r.index for r in refs] == [0, 1, 2] and [r.length for r in refs] == [12, 9, 8]
    assert refs[1].meta['tasks'] == ['place the cube']
    total = 0
    for e, r in zip(src, refs):
        g = ds.load(r.index)
        n = len(e.t)
        # the stored columns, bit for bit after the float32 cast LeRobot stores
        np.testing.assert_array_equal(g.q, e.q.astype(np.float32).astype(np.float64))
        np.testing.assert_array_equal(g.action, e.action.astype(np.float32).astype(np.float64))
        np.testing.assert_array_equal(g.t, e.t.astype(np.float32).astype(np.float64))
        np.testing.assert_array_equal(g.meta['columns']['observation.effort'], e.meta['columns']['observation.effort'])
        np.testing.assert_array_equal(g.meta['columns']['frame_index'], np.arange(n))
        np.testing.assert_array_equal(g.meta['columns']['episode_index'], np.full(n, r.index))
        np.testing.assert_array_equal(g.meta['columns']['index'], np.arange(total, total + n))
        assert ds.tasks[int(g.meta['columns']['task_index'][0])] == e.task
        total += n
        assert np.max(np.abs(g.q - e.q)) < 2.0 * 2 ** -23        # float32 rounding of values in [-2, 2]
        assert g.joint_names == e.joint_names and g.action_names == e.action_names
        assert g.task == e.task and g.robot_type == 'reference_arm' and g.fps == 10.0
        assert g.id == f'episode_{r.index:06d}' and g.index == r.index
        assert g.meta['version'] == version and g.meta['units'].startswith('as stored')
        # the camera: frame times from the video pts, the irregular episode included
        cam, c0 = g.cameras[CAM], e.cameras[CAM]
        assert (cam.width, cam.height, cam.fps) == (64, 48, 10.0)
        assert cam.intrinsics is None and cam.extrinsics is None and cam.hw_timestamps is None
        assert cam.n_frames == c0.n_frames
        assert np.max(np.abs(cam.timestamps - c0.timestamps)) <= CF.LEROBOT_TS_TOL_S
        frames = dict(cam.frames())
        sf = c0.meta['source_frames']
        assert sorted(frames) == list(range(len(sf)))
        for i, img in frames.items():
            assert img.shape == (48, 64, 3) and img.dtype == np.uint8
            p = _psnr(img, sf[i])
            assert p >= CF.LEROBOT_PSNR_DB
            for k in (i - 1, i + 1):                      # the right frame, not a neighbour
                if 0 <= k < len(sf):
                    assert p > _psnr(img, sf[k]) + 5
        # lazy subsets come back in ascending order, once each
        assert [i for i, _ in cam.frames([5, 2, 2, 0])] == [0, 2, 5]
        with pytest.raises(DS.DatasetError, match='does not exist'):
            list(cam.frames([len(sf)]))
    assert ds.truth() is None


def test_dropped_and_late_frames_keep_their_times(tmp_path, src):
    for version in VERSIONS:
        root, _ = _write(tmp_path, src, version)
        g = L.LeRobotDataset(root).load(2)
        ts = g.cameras[CAM].timestamps
        assert len(ts) == 7 and len(g.t) == 8                     # one frame fewer than rows: a drop
        assert ts[3] == pytest.approx(0.4, abs=CF.LEROBOT_TS_TOL_S)   # the gap after 0.2 s
        assert ts[5] == pytest.approx(0.613, abs=CF.LEROBOT_TS_TOL_S)  # 13 ms late
        info = json.load(open(os.path.join(root, L.INFO_PATH)))
        assert info['features'][CAM]['info']['video.fps'] == 10      # nominal, not a guess from the pts


# --- the layout, as the LeRobot source defines it ----------------------------------------------------------------

def test_v21_layout_matches_the_source(tmp_path, src):
    root, s = _write(tmp_path, src, 'v2.1')
    for j in range(3):
        assert os.path.isfile(os.path.join(root, f'data/chunk-000/episode_{j:06d}.parquet'))
        assert os.path.isfile(os.path.join(root, f'videos/chunk-000/{CAM}/episode_{j:06d}.mp4'))
    for f in ('meta/info.json', 'meta/episodes.jsonl', 'meta/tasks.jsonl', 'meta/episodes_stats.jsonl'):
        assert os.path.isfile(os.path.join(root, f))
    assert not os.path.exists(os.path.join(root, 'meta/stats.json'))   # v2.1 reads episodes_stats, not stats.json
    info = json.load(open(os.path.join(root, 'meta/info.json')))
    assert list(info) == ['codebase_version', 'robot_type', 'total_episodes', 'total_frames', 'total_tasks',
                          'total_videos', 'total_chunks', 'chunks_size', 'fps', 'splits', 'data_path', 'video_path',
                          'features']
    assert info['codebase_version'] == 'v2.1' and info['fps'] == 10 and info['chunks_size'] == 1000
    assert info['total_episodes'] == 3 and info['total_frames'] == 29 and info['total_tasks'] == 2
    assert info['total_videos'] == 3 and info['total_chunks'] == 1 and info['splits'] == {'train': '0:3'}
    assert info['data_path'] == 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet'
    assert info['video_path'] == 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'
    f = info['features']
    assert f['observation.state'] == {'dtype': 'float32', 'shape': [6], 'names': src[0].joint_names}
    assert f['observation.effort'] == {'dtype': 'float32', 'shape': [2], 'names': ['gripper', 'wrist']}
    for k, v in L.DEFAULT_FEATURES.items():
        assert f[k] == v
    assert f[CAM]['dtype'] == 'video' and f[CAM]['shape'] == [48, 64, 3]
    assert f[CAM]['names'] == ['height', 'width', 'channels']
    vi = f[CAM]['info']
    assert set(vi) == {'video.height', 'video.width', 'video.codec', 'video.pix_fmt', 'video.is_depth_map',
                       'video.fps', 'video.channels', 'has_audio'}
    assert vi['video.codec'] in ('av1', 'h264') and vi['video.pix_fmt'] == 'yuv420p' and vi['video.fps'] == 10
    eps = [json.loads(x) for x in open(os.path.join(root, 'meta/episodes.jsonl'))]
    assert eps[0] == {'episode_index': 0, 'tasks': ['pick the cube'], 'length': 12}
    tasks = [json.loads(x) for x in open(os.path.join(root, 'meta/tasks.jsonl'))]
    assert tasks == [{'task_index': 0, 'task': 'pick the cube'}, {'task_index': 1, 'task': 'place the cube'}]
    st = [json.loads(x) for x in open(os.path.join(root, 'meta/episodes_stats.jsonl'))]
    s0 = st[0]['stats']
    assert set(s0['observation.state']) == {'min', 'max', 'mean', 'std', 'count'}   # no quantiles at v2.1
    np.testing.assert_allclose(s0['observation.state']['mean'], src[0].q.astype(np.float32).mean(0), rtol=1e-6)
    assert s0['observation.state']['count'] == [12] and s0['timestamp']['min'] == [0.0]
    assert np.asarray(s0[CAM]['mean']).shape == (3, 1, 1) and s0[CAM]['count'] == [12]
    t = pq.read_table(os.path.join(root, 'data/chunk-000/episode_000000.parquet'))
    assert t.schema.field('observation.state').type == pa.list_(pa.float32(), 6)
    assert t.schema.field('timestamp').type == pa.float32() and t.schema.field('index').type == pa.int64()
    assert CAM not in t.column_names


def test_v30_layout_matches_the_source(tmp_path, src):
    root, s = _write(tmp_path, src, 'v3.0')
    assert s['files'] == ['data/chunk-000/file-000.parquet', f'videos/{CAM}/chunk-000/file-000.mp4']
    for f in ('meta/info.json', 'meta/stats.json', 'meta/tasks.parquet', 'meta/episodes/chunk-000/file-000.parquet'):
        assert os.path.isfile(os.path.join(root, f))
    assert not any(n.endswith('.jsonl') for n in os.listdir(os.path.join(root, 'meta')))
    info = json.load(open(os.path.join(root, 'meta/info.json')))
    assert list(info) == ['codebase_version', 'fps', 'features', 'total_episodes', 'total_frames', 'total_tasks',
                          'chunks_size', 'data_files_size_in_mb', 'video_files_size_in_mb', 'data_path', 'video_path',
                          'robot_type', 'splits']
    assert info['data_path'] == 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet'
    assert info['video_path'] == 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'
    assert info['data_files_size_in_mb'] == 100 and info['video_files_size_in_mb'] == 200
    vi = info['features'][CAM]['info']
    assert vi['is_depth_map'] is False and 'video.is_depth_map' not in vi
    if vi['video.codec'] == 'av1':
        assert (vi['video.g'], vi['video.crf'], vi['video.preset']) == (2, 30, 12)
    ep = pq.read_table(os.path.join(root, 'meta/episodes/chunk-000/file-000.parquet'))
    cols = set(ep.column_names)
    for c in ('episode_index', 'tasks', 'length', 'data/chunk_index', 'data/file_index', 'dataset_from_index',
              'dataset_to_index', f'videos/{CAM}/chunk_index', f'videos/{CAM}/file_index',
              f'videos/{CAM}/from_timestamp', f'videos/{CAM}/to_timestamp', 'meta/episodes/chunk_index',
              'meta/episodes/file_index', 'stats/observation.state/mean', 'stats/observation.state/q99',
              f'stats/{CAM}/mean', 'stats/timestamp/count'):
        assert c in cols, c
    rows = ep.to_pylist()
    assert [(r['dataset_from_index'], r['dataset_to_index']) for r in rows] == [(0, 12), (12, 21), (21, 29)]
    # from/to timestamps: summed durations (last frame + 1/fps) of the episodes before, as LeRobot's packing gives
    ft = [(r[f'videos/{CAM}/from_timestamp'], r[f'videos/{CAM}/to_timestamp']) for r in rows]
    np.testing.assert_allclose(ft, [(0.0, 1.2), (1.2, 2.1), (2.1, 2.9)], atol=1e-12)
    assert rows[0]['tasks'] == ['pick the cube']
    assert np.asarray(rows[0][f'stats/{CAM}/mean']).shape == (3, 1, 1)
    # tasks.parquet: 'task' restored as the pandas index (LeRobot's load_tasks uses pd.read_parquet)
    tt = pq.read_table(os.path.join(root, 'meta/tasks.parquet'))
    meta = json.loads(tt.schema.metadata[b'pandas'])
    assert meta['index_columns'] == ['task'] and tt.to_pylist() == [{'task_index': 0, 'task': 'pick the cube'},
                                                                    {'task_index': 1, 'task': 'place the cube'}]
    st = json.load(open(os.path.join(root, 'meta/stats.json')))
    assert set(st['observation.state']) == {'min', 'max', 'mean', 'std', 'count', 'q01', 'q10', 'q50', 'q90', 'q99'}
    allq = np.concatenate([e.q.astype(np.float32) for e in src])
    np.testing.assert_allclose(st['observation.state']['mean'], allq.astype(np.float64).mean(0), rtol=1e-9)
    np.testing.assert_allclose(st['observation.state']['std'], allq.astype(np.float64).std(0), rtol=1e-9)
    assert st['observation.state']['count'] == [29] and st[CAM]['count'] == [28]
    # one row group per episode (dataset_writer.py)
    assert pq.ParquetFile(os.path.join(root, 'data/chunk-000/file-000.parquet')).num_row_groups == 3
    ds = L.LeRobotDataset(root)
    assert ds.stats()['observation.state']['count'] == [29]
    assert ds.episode_stats(1)['observation.state']['count'] == [9]


def test_v21_aggregated_stats_match_numpy(tmp_path, src):
    root, _ = _write(tmp_path, src, 'v2.1')
    st = L.LeRobotDataset(root).stats()
    allq = np.concatenate([e.q.astype(np.float32) for e in src]).astype(np.float64)
    np.testing.assert_allclose(st['observation.state']['mean'], allq.mean(0), rtol=1e-9)
    np.testing.assert_allclose(st['observation.state']['std'], allq.std(0), rtol=1e-9)
    np.testing.assert_allclose(st['observation.state']['min'], allq.min(0))


def test_v30_packed_files_roll_over(tmp_path, src):
    root, s = _write(tmp_path, src, 'v3.0', data_file_size_mb=1e-6, video_file_size_mb=1e-6)
    assert 'data/chunk-000/file-002.parquet' in s['files'] and f'videos/{CAM}/chunk-000/file-002.mp4' in s['files']
    rows = pq.read_table(os.path.join(root, 'meta/episodes/chunk-000/file-000.parquet')).to_pylist()
    assert [r['data/file_index'] for r in rows] == [0, 1, 2]
    assert [r[f'videos/{CAM}/from_timestamp'] for r in rows] == [0.0, 0.0, 0.0]   # each starts a new file
    ds = L.LeRobotDataset(root)
    for e in src:
        g = ds.load(e.index)
        np.testing.assert_array_equal(g.q, e.q.astype(np.float32).astype(np.float64))
        cam = g.cameras[CAM]
        assert np.max(np.abs(cam.timestamps - e.cameras[CAM].timestamps)) <= CF.LEROBOT_TS_TOL_S
        f = dict(cam.frames([0, cam.n_frames - 1]))
        assert _psnr(f[0], e.cameras[CAM].meta['source_frames'][0]) >= CF.LEROBOT_PSNR_DB


def test_h264_on_request_is_labelled(tmp_path):
    root = str(tmp_path / 'h264')
    s = L.write_lerobot(root, [_episode()], 'v3.0', 10, codec='h264')
    assert s['codec'] == 'H.264 (libx264)' and s['encoder'] == 'libx264'
    g = L.LeRobotDataset(root).load(0)
    assert g.cameras[CAM].meta['codec'] == 'h264'
    assert _psnr(dict(g.cameras[CAM].frames([3]))[3], CF.lerobot_pattern(3)) >= CF.LEROBOT_PSNR_DB
    with pytest.raises(DS.DatasetError, match='not one the LeRobot writer offers'):
        L.available_codec('vp9')


def test_dataset_without_cameras_or_actions(tmp_path):
    e = E.Episode('e', 0, np.arange(5) / 10, np.ones((5, 3)), ['x', 'y', 'z'], fps=10.0, task='still')
    for v in VERSIONS:
        root = str(tmp_path / f'plain_{v}')
        s = L.write_lerobot(root, [e], v)
        assert s['codec'] is None
        info = json.load(open(os.path.join(root, L.INFO_PATH)))
        assert info['video_path'] is None and 'action' not in info['features']
        g = L.LeRobotDataset(root).load(0)
        assert g.action is None and g.cameras == {} and g.joint_names == ['x', 'y', 'z']


# --- reading what others wrote ------------------------------------------------------------------------------------

def test_hub_snapshot_folders(tmp_path, src):
    root, _ = _write(tmp_path, src, 'v3.0')
    repo = tmp_path / 'hub' / 'datasets--someorg--somedata'
    snap = repo / 'snapshots' / '0123abcd'
    shutil.copytree(root, snap)
    (repo / 'refs').mkdir()
    (repo / 'refs' / 'main').write_text('0123abcd')
    for p in (repo, snap, snap / 'meta' / 'info.json'):
        ds = DS.open_dataset(str(p))
        assert ds.format == 'lerobot' and ds.root == str(snap) and len(ds.episodes()) == 3


def test_names_as_dict_and_image_columns(tmp_path):
    cv2 = pytest.importorskip('cv2', reason='opencv-python (a core dependency) does not import here')
    root = str(tmp_path / 'img')
    e = E.Episode('e', 0, np.arange(4) / 10, np.arange(8.0).reshape(4, 2), ['a', 'b'], fps=10.0, task='x')
    L.write_lerobot(root, [e], 'v2.1')
    # an older-style dataset: names as {'motors': [...]} and a camera stored as PNG bytes in the Parquet
    info = json.load(open(os.path.join(root, L.INFO_PATH)))
    info['features']['observation.state']['names'] = {'motors': ['shoulder', 'elbow']}
    info['features']['observation.images.top'] = {'dtype': 'image', 'shape': [48, 64, 3],
                                                  'names': ['height', 'width', 'channels']}
    json.dump(info, open(os.path.join(root, L.INFO_PATH), 'w'))
    dp = os.path.join(root, 'data/chunk-000/episode_000000.parquet')
    t = pq.read_table(dp)
    pngs = [{'bytes': cv2.imencode('.png', CF.lerobot_pattern(i)[:, :, ::-1])[1].tobytes(), 'path': None}
            for i in range(4)]
    t = t.append_column('observation.images.top', pa.array(pngs))
    pq.write_table(t, dp)
    g = L.LeRobotDataset(root).load(0)
    assert g.joint_names == ['shoulder', 'elbow']
    cam = g.cameras['observation.images.top']
    assert cam.n_frames == 4 and cam.meta['source'] == 'parquet images'
    for i, img in cam.frames():
        np.testing.assert_array_equal(img, CF.lerobot_pattern(i))     # PNG is lossless


@pytest.mark.parametrize('version', ['v2.0', 'v1.6', 'v3.1', 'v4.0'])
def test_refuses_other_versions_naming_the_supported_ones(tmp_path, version):
    root = tmp_path / f'old_{version}'
    (root / 'meta').mkdir(parents=True)
    (root / 'meta' / 'info.json').write_text(json.dumps({'codebase_version': version, 'fps': 30, 'features': {}}))
    with pytest.raises(DS.DatasetError, match=r'codebase version .* reads versions v2\.1 and v3\.0 only'):
        DS.open_dataset(str(root))


def test_refuses_unstated_version_and_other_storage(tmp_path):
    root = tmp_path / 'nover'
    (root / 'meta').mkdir(parents=True)
    (root / 'meta' / 'info.json').write_text(json.dumps({'fps': 30, 'features': {}}))
    with pytest.raises(DS.DatasetError, match='does not state its LeRobot codebase_version'):
        DS.open_dataset(str(root))
    (root / 'meta' / 'info.json').write_text(json.dumps({'codebase_version': 'v3.0', 'fps': 30, 'features': {},
                                                         'storage_format': 'lance'}))
    with pytest.raises(DS.DatasetError, match='Parquet and MP4 layout only'):
        DS.open_dataset(str(root))
    assert not L.sniff(str(tmp_path / 'nowhere'))


def test_missing_files_are_plain_errors(tmp_path, src):
    root, _ = _write(tmp_path, src, 'v2.1')
    os.remove(os.path.join(root, f'videos/chunk-000/{CAM}/episode_000001.mp4'))
    ds = L.LeRobotDataset(root)
    with pytest.raises(DS.DatasetError, match='has no video for camera'):
        ds.load(1)
    with pytest.raises(DS.DatasetError, match='has no episode 7'):
        ds.load(7)
    os.remove(os.path.join(root, 'meta/tasks.jsonl'))
    with pytest.raises(DS.DatasetError, match='has no meta/tasks.jsonl'):
        L.LeRobotDataset(root)


# --- the sidecar and the writer's refusals -------------------------------------------------------------------------

def test_truth_sidecar_round_trip(tmp_path):
    truth = {'schema': L.TRUTH_SCHEMA, 'project_digest': 'abc', 'generator': {'seed': 1}, 'cameras': {},
             'episodes': {'0': {'faults': [], 'extrinsics': {}, 'visibility': {}}}}
    root = str(tmp_path / 'synthetic')
    L.write_lerobot(root, [_episode()], 'v3.0', truth=truth)
    assert DS.open_dataset(root).truth() == truth
    with pytest.raises(DS.DatasetError, match='dexvantage.truth/1'):
        L.write_lerobot(str(tmp_path / 'bad'), [_episode()], truth={'schema': 'other'})


def test_writer_refusals_are_plain_sentences(tmp_path):
    p = str(tmp_path / 'w')
    with pytest.raises(DS.DatasetError, match='whole number'):
        L.write_lerobot(p, [_episode()], fps=29.97)
    with pytest.raises(DS.DatasetError, match='even width and height'):
        L.write_lerobot(p, [_episode(size=(47, 64))])
    assert not os.path.exists(p)                               # a failed write leaves nothing behind
    with pytest.raises(DS.DatasetError, match='writes versions v2.1 and v3.0'):
        L.write_lerobot(p, [_episode()], version='v2.0')
    with pytest.raises(DS.DatasetError, match='no episodes'):
        L.write_lerobot(p, [], fps=10)
    with pytest.raises(DS.DatasetError, match='contains "/"'):
        L.write_lerobot(p, [_episode(key='observation/images/x')])
    a, b = _episode(), _episode(key='observation.images.other')
    with pytest.raises(DS.DatasetError, match='same cameras'):
        L.write_lerobot(p, [a, b])
    with pytest.raises(DS.DatasetError, match='must start at 0 s or later and increase'):
        L.write_lerobot(p, [_episode(ts=[0.0, 0.2, 0.1, 0.3, 0.4, 0.5])])
    with pytest.raises(DS.DatasetError, match='no column observation.effort'):
        L.write_lerobot(p, [_episode()], features=CF.LEROBOT_EXTRA)
    other = tmp_path / 'notes'
    other.mkdir()
    (other / 'keep.txt').write_text('mine')
    with pytest.raises(DS.DatasetError, match='not empty'):
        L.write_lerobot(str(other), [_episode()])
    with pytest.raises(DS.DatasetError, match='is not a LeRobot dataset, so it is not overwritten'):
        L.write_lerobot(str(other), [_episode()], overwrite=True)
    assert (other / 'keep.txt').read_text() == 'mine'
    L.write_lerobot(p, [_episode()])
    L.write_lerobot(p, [_episode(n=4)], overwrite=True)        # an existing dataset may be replaced
    assert L.LeRobotDataset(p).episodes()[0].length == 4


def test_missing_optional_packages_say_what_to_install(monkeypatch, tmp_path):
    real = builtins.__import__

    def fake(name, *a, **k):
        if name.startswith('pyarrow'):
            raise ImportError(name)
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, '__import__', fake)
    with pytest.raises(DS.DatasetError, match=r'pip install "dexvantage\[lerobot\]"'):
        L.write_lerobot(str(tmp_path / 'x'), [_episode()])


# --- the validation check --------------------------------------------------------------------------------------------

def test_validation_check_passes():
    from dexvantage.validation import suite
    spec = next(s for s in suite.CHECKS if s.id == 'formats.lerobot_roundtrip')
    assert spec.model == 'formats' and set(spec.refs) <= set(__import__(
        'dexvantage.validation.references', fromlist=['REFERENCES']).REFERENCES)
    ctx = suite.Ctx(None, None, None, suite.ValidationSettings())
    try:
        c = suite.run_one(spec, ctx)
    finally:
        ctx.close()
    assert c.passed, (c.exception, c.detail.get('mismatches'))
    for v in VERSIONS:
        assert c.detail[v]['psnr_reader_min_db'] >= CF.LEROBOT_PSNR_DB and c.detail[v]['order_margin_db'] > 0
    assert 'reads' in c.note or 'Tolerances' in c.note


@pytest.mark.parametrize('codec', ['av1', 'h264'])
def test_a_late_camera_keeps_its_offset_inside_a_packed_file(tmp_path, codec):
    """ffmpeg's concat demuxer would shift a late-starting episode back to its from_timestamp; the remux must not."""
    a = _episode(n=5)
    b = _episode(n=6, ts=np.arange(6) / 10 + 0.03)                # this camera starts 30 ms after the rows
    c = _episode(n=4)
    root = str(tmp_path / f'late_{codec}')
    L.write_lerobot(root, [a, b, c], 'v3.0', 10, codec=codec)
    ds = L.LeRobotDataset(root)
    for k, e in enumerate((a, b, c)):
        cam = ds.load(k).cameras[CAM]
        assert np.max(np.abs(cam.timestamps - e.cameras[CAM].timestamps)) <= CF.LEROBOT_TS_TOL_S
        assert cam.meta['file'] == f'videos/{CAM}/chunk-000/file-000.mp4'
        frames = dict(cam.frames())
        assert all(_psnr(frames[i], CF.lerobot_pattern(i)) >= CF.LEROBOT_PSNR_DB for i in frames)
    rows = pq.read_table(os.path.join(root, 'meta/episodes/chunk-000/file-000.parquet')).to_pylist()
    lo = [r[f'videos/{CAM}/from_timestamp'] for r in rows]
    np.testing.assert_allclose(lo, [0.0, 0.5, 0.5 + 0.53 + 0.1], atol=CF.LEROBOT_TS_TOL_S)
