"""Regression tests from the correctness review of Phases 4, 5 and 8: the result cache against a recorded dataset that
changes on disk and against the new project sections (data, rig sync, lights, teleop devices); and the conventions
the review checked and found right, pinned against independent references so a later change cannot flip them: the
lens remap direction, LeRobot frame seeking in v2.1 and v3.0 files, CDR alignment and byte order, TFRecord's
CRC-32C, ACT's ensembling order and the blind gap."""
import math
import struct

import numpy as np
import pytest

from dexvantage.core import jobs
from dexvantage.core.project import data_from_dict, rig_from_dict
from dexvantage.core.robot.samples import sample_project


# ---------------------------------------------------------------------- the result cache
def _touch(path, text):
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(text)


@pytest.mark.parametrize('sid', jobs.DATASET_STUDIES)
def test_cache_key_follows_a_dataset_that_changes_on_disk(tmp_path, sid):
    """The project names a dataset by its path only; a dataset re-recorded or re-generated at the same path used to
    hit the cache (the silhouette page passed no fingerprint at all) and showed the old result."""
    ds = tmp_path / 'ds'
    (ds / 'meta').mkdir(parents=True)
    _touch(ds / 'meta' / 'info.json', '{"a": 1}')
    p = sample_project('so100')
    if sid == 'silhouette_calib':
        p.analysis.studies[sid] = {'dataset': str(ds)}
    else:
        p.data = data_from_dict({'datasets': [{'name': 'd', 'path': str(ds)}]})
    c = jobs.ResultCache()
    k0 = c.key(sid, p)
    assert c.key(sid, p) == k0                                   # stable while nothing changes
    _touch(ds / 'meta' / 'info.json', '{"a": 12}')               # same name, new size
    k1 = c.key(sid, p)
    assert k1 != k0
    _touch(ds / 'meta' / 'episodes.jsonl', '{}')                 # a new file
    assert c.key(sid, p) != k1
    # a study that reads no dataset keeps its key; a dataset that is gone gives a key, not an error
    assert jobs.dataset_fingerprint('lighting', p) == ''
    if sid == 'silhouette_calib':
        p.analysis.studies[sid] = {'dataset': str(tmp_path / 'gone')}
    else:
        p.data = data_from_dict({'datasets': [{'name': 'd', 'path': str(tmp_path / 'gone')}]})
    assert c.key(sid, p)[-1] == ''


def test_cache_key_follows_the_new_sections():
    """Lights, teleop devices, the rig's sync block and the data section are project content: each change misses."""
    base = jobs.ResultCache().key('lighting', sample_project('so100'))

    def changed(fn):
        p = sample_project('so100')
        fn(p)
        return jobs.ResultCache().key('lighting', p) != base

    assert changed(lambda p: p.workspace.lights.append({'uid': 'L1', 'name': 'lamp', 'kind': 'point',
                                                        'pose': {'position': [0, 0, 1], 'quaternion': [0, 0, 0, 1]}}))
    assert changed(lambda p: p.tasks.teleop_devices.append({'uid': 'T1', 'kind': 'vr'}))
    assert changed(lambda p: setattr(p, 'rig', rig_from_dict(
        {**(p.rig or {}), 'sync': {'clocks': [{'uid': 'c1', 'name': 'ptp', 'kind': 'ptp'}], 'streams': {}}})))
    assert changed(lambda p: setattr(p, 'data', data_from_dict({'requirements': {'max_offset_ms': 10}})))


# ---------------------------------------------------------------------- lens remap direction (core/sim/render.py)
def test_remap_maps_each_lens_pixel_to_the_base_pixel_its_ray_lands_on():
    """Independent of unproject: the base pixel a lens pixel samples, cast back as a ray of the base pinhole and
    projected through the lens model (forward, CameraModel.project), lands on that lens pixel."""
    from dexvantage.core.optics import CameraModel
    from dexvantage.core.sim import render as RD
    cams = (CameraModel(160, 120, 130.0, 128.0, 80.6, 59.1, 'brown', (-0.25, 0.08, 0.001, -0.0008, -0.01)),
            CameraModel(160, 120, 45.0, 45.0, 79.5, 59.5, 'kb', (0.02, -0.01, 0.003, -0.0005), math.radians(100)))
    for cam, path in zip(cams, ('wide', 'cube')):
        plan = RD.lens_plan(cam)
        assert plan.path == path
        for s in plan.subs:
            vv, uu = np.nonzero(s.sel)
            mu, mv = s.map_u[vv, uu].astype(float), s.map_v[vv, uu].astype(float)
            K = s.K
            assert mu.min() >= -0.5 and mu.max() <= K.width - 0.5 and mv.min() >= -0.5 and mv.max() <= K.height - 0.5
            rb = np.stack([(mu - K.cx) / K.fx, (mv - K.cy) / K.fy, np.ones_like(mu)], -1) @ s.R.T
            uv, ok = cam.project(rb)
            assert np.asarray(ok, bool).all()
            assert np.max(np.hypot(uv[:, 0] - uu, uv[:, 1] - vv)) < 1e-3


# ---------------------------------------------------------------------- LeRobot frame seeking
def _colour(i):
    return ((i * 37) % 256, (i * 91) % 256, (i * 13 + 50) % 256)


def _episode(j, n, ts):
    from dexvantage.core.datasets import episode as E
    frames = [np.full((48, 64, 3), _colour(100 * j + i), np.uint8) for i in range(len(ts))]

    def dec(idx):
        for i in (range(len(frames)) if idx is None else idx):
            yield i, frames[i]
    cam = E.CameraStream('observation.images.c', 64, 48, 10.0, np.asarray(ts, float), _decode=dec)
    return E.Episode(f'e{j}', j, np.arange(n) / 10.0, np.zeros((n, 2)), ['a', 'b'], fps=10.0, task='t',
                     cameras={'observation.images.c': cam})


@pytest.mark.parametrize('version', ('v2.1', 'v3.0'))
@pytest.mark.parametrize('irregular', (False, True))
def test_lerobot_random_access_frames_are_the_right_frames(tmp_path, version, irregular):
    """Any subset of an episode's frames decodes to exactly those frames: the first, the last, the first frame of a
    later episode in a packed v3.0 file (the seek lands in the episode before it), and frames across a drop."""
    pytest.importorskip('pyarrow', reason='pyarrow (the lerobot extra) is not installed')
    pytest.importorskip('av', reason='av (the lerobot extra) is not installed')
    from dexvantage.core import datasets as DS
    from dexvantage.core.datasets import lerobot as L
    eps, want_ts = [], []
    for j, n in enumerate((14, 6, 17)):
        ts = np.arange(n) / 10.0
        if irregular:
            ts = np.delete(ts, [3, 4]) + 0.013                  # two dropped frames and a late camera
        eps.append(_episode(j, n, ts))
        want_ts.append(ts)
    root = str(tmp_path / f'ds_{version}')
    L.write_lerobot(root, eps, version, 10, 'x')
    with DS.open_dataset(root) as ds:
        for j, ref in enumerate(ds.episodes()):
            c = ds.load(ref.index).cameras['observation.images.c']
            assert len(c.timestamps) == len(want_ts[j])
            assert np.max(np.abs(c.timestamps - want_ts[j])) < 1e-4
            m = len(want_ts[j])
            for sub in ([0], [m - 1], [m // 2], [1, m - 2], list(range(0, m, 3))):
                got = dict(c.frames(sub))
                assert sorted(got) == sorted(set(sub))
                for i, img in got.items():
                    assert np.abs(img.astype(int) - np.array(_colour(100 * j + i))).mean() < 6, (j, i)


# ---------------------------------------------------------------------- CDR (core/rosbag.py)
def _cdr(endian, fields):
    body = bytearray()

    def al(n):
        while len(body) % n:
            body.append(0)
    for kind, v in fields:
        if kind == 'str':
            s = v.encode() + b'\x00'
            al(4)
            body.extend(struct.pack(endian + 'I', len(s)) + s)
        elif kind == 'u32':
            al(4)
            body.extend(struct.pack(endian + 'I', v))
        elif kind == 'i32':
            al(4)
            body.extend(struct.pack(endian + 'i', v))
        elif kind == 'f64s':
            al(4)
            body.extend(struct.pack(endian + 'I', len(v)))
            if v:
                al(8)
                body.extend(struct.pack(endian + f'{len(v)}d', *v))
        elif kind == 'strs':
            al(4)
            body.extend(struct.pack(endian + 'I', len(v)))
            for s in v:
                b = s.encode() + b'\x00'
                al(4)
                body.extend(struct.pack(endian + 'I', len(b)) + b)
    head = b'\x00\x01\x00\x00' if endian == '<' else b'\x00\x00\x00\x00'
    return bytes(head + body)


@pytest.mark.parametrize('endian', ('<', '>'))
def test_joint_state_cdr_hand_encoded(endian):
    """A JointState laid out by hand (XCDR1: every primitive aligned to its size from the payload start, the 8-byte
    float64 elements to 8 after the 4-byte count; 00 01 little-endian, 00 00 big-endian)."""
    from dexvantage.core import rosbag as RB
    data = _cdr(endian, [('i32', 5), ('u32', 7), ('str', 'base'), ('strs', ['j1', 'joint2']),
                         ('f64s', [1.5, -2.25]), ('f64s', []), ('f64s', [3.0])])
    m = RB.decode('sensor_msgs/msg/JointState', data)
    assert m['header'] == {'stamp': {'sec': 5, 'nanosec': 7}, 'frame_id': 'base'}
    assert m['name'] == ['j1', 'joint2'] and m['position'] == [1.5, -2.25] and m['velocity'] == []
    assert m['effort'] == [3.0]
    if endian == '<':
        assert RB.encode('sensor_msgs/msg/JointState', m) == data


# ---------------------------------------------------------------------- TFRecord CRC-32C
def test_crc32c_rfc3720_vectors_and_tfrecord_mask():
    """RFC 3720 B.4 test vectors and TFRecord's mask (rotate right 15, add 0xa282ead8)."""
    from dexvantage.core.datasets import rlds as R
    assert R.crc32c(b'123456789') == 0xE3069283
    assert R.crc32c(bytes(32)) == 0x8A9136AA
    assert R.crc32c(b'\xff' * 32) == 0x62A8AB43
    assert R.crc32c(bytes(range(32))) == 0x46DD794E
    assert R.crc32c(bytes(range(31, -1, -1))) == 0x113FDB5C
    big = bytes(np.random.default_rng(3).integers(0, 256, 100_003, dtype=np.uint8))
    assert R.crc32c(big[40_000:], R.crc32c(big[:40_000])) == R.crc32c(big)
    c = 0x12345678
    assert R.mask_crc(c) == ((((c >> 15) | (c << 17)) & 0xFFFFFFFF) + 0xA282EAD8) & 0xFFFFFFFF
    assert R.unmask_crc(R.mask_crc(0xDEADBEEF)) == 0xDEADBEEF


# ---------------------------------------------------------------------- ACT temporal ensembling order
def test_act_ensembler_matches_the_official_code_order():
    """ACT's imitate_episodes.py: all_time_actions[[t], t:t+H] = chunk; the rows populated for step t are taken in
    issue order (row 0 the OLDEST) and weighted exp(-k·arange(n)). ActEnsembler must give the same action."""
    from dexvantage.core.studies import action_chunk_sync as AC
    rng = np.random.default_rng(1)
    T, H, k = 12, 5, 0.3
    chunks = rng.normal(size=(T, H, 2)) + 3.0                     # never 0: the official code tests != 0
    allt = np.zeros((T, T + H, 2))
    ens = AC.ActEnsembler(H, k)
    for t in range(T):
        allt[t, t:t + H] = chunks[t]
        ens.add(t, chunks[t])
        cur = allt[:, t]
        cur = cur[np.all(cur != 0, axis=1)]
        w = np.exp(-k * np.arange(len(cur)))
        w /= w.sum()
        assert np.allclose(ens.action(t), (cur * w[:, None]).sum(axis=0), atol=1e-12)
    # m > 0 favours the oldest prediction: the mean age grows with m
    assert AC.ensemble_mean_age(0.5, 8) > AC.ensemble_mean_age(0.0, 8) > AC.ensemble_mean_age(-0.5, 8)
    assert math.isclose(AC.ensemble_mean_age(0.0, 8), 3.5)


# ---------------------------------------------------------------------- the blind gap
def test_blind_gap_runs_from_contact_to_the_first_seen_distance():
    from dexvantage.core.studies.dex_hand_tactile import gap_from_samples
    d = [0.0, 0.01, 0.02, 0.03, 0.05]
    assert gap_from_samples(d, [False, False, True, True, True])[:2] == (pytest.approx(0.015), False)
    assert gap_from_samples(d, [True, False, False, True, True])[:2] == (0.0, False)     # seen at contact
    assert gap_from_samples(d, [False] * 5)[:2] == (0.05, True)                          # never seen: "or more"
    # the order of the samples does not matter
    assert gap_from_samples(d[::-1], [True, True, True, False, False])[0] == pytest.approx(0.015)
