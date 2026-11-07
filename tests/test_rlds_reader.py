"""The RLDS / Open X-Embodiment reader (core/datasets/rlds.py): CRC-32C against RFC 3720's published vectors and a
bitwise reference, the hand-written protobuf parser against Google's protobuf runtime (when installed) and against an
independent encoder (tests/tfrecord_writer.py), TFDS datasets written the way TFDS lays them out, SequenceExamples,
damaged files and the formats.tfrecord_crc check."""
import gzip
import json
import os
import shutil
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tfrecord_writer as TW  # noqa: E402

from dexvantage.core import datasets as DS  # noqa: E402
from dexvantage.core.datasets import rlds as R  # noqa: E402

RFC3720 = [(bytes(32), 0x8A9136AA), (b'\xff' * 32, 0x62A8AB43), (bytes(range(32)), 0x46DD794E),
           (bytes(range(31, -1, -1)), 0x113FDB5C),
           (bytes.fromhex('01c00000 00000000 00000000 00000000 14000000 00000400 00000014 00000018 28000000 '
                          '00000000 02000000 00000000'.replace(' ', '')), 0xD9963A56)]


@pytest.fixture
def tmp(tmp_path):
    yield tmp_path
    shutil.rmtree(tmp_path, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------------------------
# CRC-32C and framing

def test_crc32c_published_vectors():
    for data, crc in RFC3720:
        assert R.crc32c(data) == crc == TW.crc32c_bitwise(data)
    assert R.crc32c(b'123456789') == 0xE3069283                      # the CRC catalogue's check value


@pytest.mark.parametrize('n', [0, 1, 7, 8191, 8192, 9000, 65537, 400_001])
def test_crc32c_matches_bitwise_and_continues(n):
    data = np.random.default_rng(n).integers(0, 256, n, dtype=np.uint8).tobytes()
    ref = TW.crc32c_bitwise(data) if n < 70_000 else None
    got = R.crc32c(data)
    if ref is not None:
        assert got == ref
    cut = n // 3
    assert R.crc32c(data[cut:], R.crc32c(data[:cut])) == got
    assert R.crc32c(bytearray(data)) == got and R.crc32c(memoryview(data)) == got


def test_mask_round_trip_and_tf_formula():
    for c in (0, 1, 0xFFFFFFFF, 0x12345678, 0xE3069283):
        assert R.mask_crc(c) == TW.masked(c) and R.unmask_crc(R.mask_crc(c)) == c


def test_tfrecord_framing_and_damage(tmp):
    recs = [b'', b'a', bytes(range(256)) * 50, b'last']
    p = str(tmp / 'x.tfrecord')
    TW.write_tfrecord(p, recs)
    assert R.read_tfrecords(p) == recs
    blob = bytearray(open(p, 'rb').read())
    bad = bytearray(blob)
    bad[12 + 0 + 4 + 12 + 1 + 4 + 12 + 100] ^= 1                       # inside record 3's data
    open(p, 'wb').write(bad)
    f = R.TFRecordFile(p)
    assert len(f.index()) == 4                                        # headers are fine
    with pytest.raises(DS.DatasetError, match='Record 3 .*checksum'):
        list(f.records())
    bad = bytearray(blob)
    bad[2] ^= 0xFF                                                    # the first length
    open(p, 'wb').write(bad)
    with pytest.raises(DS.DatasetError, match='length of record 1'):
        R.TFRecordFile(p).index()
    open(p, 'wb').write(blob[:-7])
    with pytest.raises(DS.DatasetError, match='truncated'):
        list(R.TFRecordFile(p).records())
    gz = str(tmp / 'x.tfrecord.gz')
    TW.write_tfrecord(gz, recs, compress=True)
    assert R.read_tfrecords(gz) == recs


# ---------------------------------------------------------------------------------------------------------------------
# protobuf

def _pb_classes():
    """tf.train.Example / SequenceExample classes built at run time from feature.proto and example.proto's
    definitions (tensorflow/core/example), with Google's protobuf runtime: no TensorFlow."""
    pytest.importorskip('google.protobuf')
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    F = descriptor_pb2.FieldDescriptorProto
    fd = descriptor_pb2.FileDescriptorProto(name='dv_example_test.proto', package='dvtest', syntax='proto3')

    def msg(name, fields, map_entry=False):
        m = fd.message_type.add(name=name)
        if map_entry:
            m.options.map_entry = True
        for num, fname, label, ftype, tname, packed in fields:
            f = m.field.add(name=fname, number=num, label=label, type=ftype)
            if tname:
                f.type_name = tname
            if packed:
                f.options.packed = True
        return m
    REP, OPT = F.LABEL_REPEATED, F.LABEL_OPTIONAL
    msg('BytesList', [(1, 'value', REP, F.TYPE_BYTES, None, False)])
    msg('FloatList', [(1, 'value', REP, F.TYPE_FLOAT, None, True)])
    msg('Int64List', [(1, 'value', REP, F.TYPE_INT64, None, True)])
    feat = msg('Feature', [(1, 'bytes_list', OPT, F.TYPE_MESSAGE, '.dvtest.BytesList', False),
                           (2, 'float_list', OPT, F.TYPE_MESSAGE, '.dvtest.FloatList', False),
                           (3, 'int64_list', OPT, F.TYPE_MESSAGE, '.dvtest.Int64List', False)])
    feat.oneof_decl.add(name='kind')
    for f in feat.field:
        f.oneof_index = 0
    fs = msg('Features', [(1, 'feature', REP, F.TYPE_MESSAGE, '.dvtest.Features.FeatureEntry', False)])
    e = fs.nested_type.add(name='FeatureEntry')
    e.options.map_entry = True
    e.field.add(name='key', number=1, label=OPT, type=F.TYPE_STRING)
    e.field.add(name='value', number=2, label=OPT, type=F.TYPE_MESSAGE, type_name='.dvtest.Feature')
    msg('FeatureList', [(1, 'feature', REP, F.TYPE_MESSAGE, '.dvtest.Feature', False)])
    fl = msg('FeatureLists', [(1, 'feature_list', REP, F.TYPE_MESSAGE, '.dvtest.FeatureLists.FeatureListEntry',
                               False)])
    e = fl.nested_type.add(name='FeatureListEntry')
    e.options.map_entry = True
    e.field.add(name='key', number=1, label=OPT, type=F.TYPE_STRING)
    e.field.add(name='value', number=2, label=OPT, type=F.TYPE_MESSAGE, type_name='.dvtest.FeatureList')
    msg('Example', [(1, 'features', OPT, F.TYPE_MESSAGE, '.dvtest.Features', False)])
    msg('SequenceExample', [(1, 'context', OPT, F.TYPE_MESSAGE, '.dvtest.Features', False),
                            (2, 'feature_lists', OPT, F.TYPE_MESSAGE, '.dvtest.FeatureLists', False)])
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fd)
    get = message_factory.GetMessageClass
    return get(pool.FindMessageTypeByName('dvtest.Example')), get(pool.FindMessageTypeByName('dvtest.SequenceExample'))


def test_parser_against_google_protobuf():
    Example, SequenceExample = _pb_classes()
    rng = np.random.default_rng(3)
    ex = Example()
    floats = rng.normal(size=257).astype(np.float32)
    ints = [0, 1, -1, 2 ** 62, -(2 ** 63), 300, 2 ** 63 - 1]
    ex.features.feature['steps/observation/state'].float_list.value.extend(floats.tolist())
    ex.features.feature['ints'].int64_list.value.extend(ints)
    ex.features.feature['names'].bytes_list.value.extend([b'', b'\x00\xff', 'é'.encode()])
    ex.features.feature['empty'].float_list.SetInParent()
    got = R.parse_example(ex.SerializeToString())
    assert got['steps/observation/state'][0] == 'float' and np.array_equal(got['steps/observation/state'][1], floats)
    assert got['ints'] == ('int64', got['ints'][1]) and got['ints'][1].tolist() == ints
    assert got['names'] == ('bytes', [b'', b'\x00\xff', 'é'.encode()])
    assert got['empty'][0] == 'float' and got['empty'][1].size == 0
    assert not R.is_sequence_example(ex.SerializeToString())
    # the test writer's bytes read by Google's runtime: the fixtures are well formed
    mine = TW.example({'a': TW.feature('float', [1.5, -2.0], packed=False), 'b': TW.feature('int64', [-5, 7]),
                       'c': TW.feature('bytes', [b'xy'])})
    back = Example.FromString(mine)
    assert list(back.features.feature['a'].float_list.value) == [1.5, -2.0]
    assert list(back.features.feature['b'].int64_list.value) == [-5, 7]
    assert list(back.features.feature['c'].bytes_list.value) == [b'xy']
    se = SequenceExample()
    se.context.feature['episode_id'].bytes_list.value.append(b'ep7')
    for i in range(4):
        se.feature_lists.feature_list['observation/state'].feature.add().float_list.value.extend([i, i + 0.5])
        se.feature_lists.feature_list['is_first'].feature.add().int64_list.value.append(int(i == 0))
    data = se.SerializeToString()
    assert R.is_sequence_example(data)
    ctx, lists = R.parse_sequence_example(data)
    assert ctx['episode_id'] == ('bytes', [b'ep7'])
    assert [v[1].tolist() for v in lists['observation/state']] == [[i, i + 0.5] for i in range(4)]
    assert [int(v[1][0]) for v in lists['is_first']] == [1, 0, 0, 0]


def test_parser_rejects_garbage():
    with pytest.raises(DS.DatasetError, match='not a readable'):
        R.parse_example(b'\x0a\xff\xff\xff\xff\x0f')                  # a length far past the end
    with pytest.raises(DS.DatasetError):
        R.parse_example(b'\x0b')                                      # wire type 3 (groups) is not used


# ---------------------------------------------------------------------------------------------------------------------
# datasets

def test_oxe_dataset_round_trip(tmp):
    cv2 = pytest.importorskip('cv2')
    folder, sent = TW.write_oxe(str(tmp), 'toy_oxe', {'train': [[5, 3], [4]], 'test': [[6]]}, zlib_extra=True)
    assert cv2 is not None
    for target in (folder, str(tmp / 'toy_oxe')):                     # the version folder or its parent
        ds = DS.open_dataset(target)
        assert ds.format == 'rlds' and ds.version == '0.1.0' and isinstance(ds, R.RldsDataset)
        refs = ds.episodes()
        assert [r.id for r in refs] == ['train-000000', 'train-000001', 'train-000002', 'test-000000']
    order = sent['train'] + sent['test']
    for i, s in enumerate(order):
        ep = ds.load(i)
        n = s['n']
        assert ep.id == refs[i].id and len(ep.t) == n and ep.q.shape == (n, 7)
        assert np.array_equal(ep.q, s['state'].astype(np.float64))
        assert ep.joint_names == [f'state_{k}' for k in range(7)] and ep.meta['state_feature'] == 'observation/state'
        exp_action = np.concatenate([s['world_vector'], s['rotation_delta'], s['open_gripper'][:, None]], axis=1)
        assert ep.action.shape == (n, 7) and np.allclose(ep.action, exp_action.astype(np.float32))
        assert ep.action_names == ['world_vector_0', 'world_vector_1', 'world_vector_2', 'rotation_delta_0',
                                   'rotation_delta_1', 'rotation_delta_2', 'open_gripper']
        assert ep.task == s['task']
        assert np.array_equal(ep.t, np.arange(n)) and ep.meta['time_base'] == 'steps' and ep.fps is None
        assert ep.meta['rlds']['is_first'].tolist() == [True] + [False] * (n - 1) and not ep.meta['warnings']
        assert ep.meta['episode_metadata'] == {'file_path': 'episode_raw/ep.npy'}
        assert set(ep.cameras) == {'observation/image', 'observation/wrist_image'}
        assert set(ep.meta['depth_streams']) == {'observation/depth'}
        cam = ep.cameras['observation/wrist_image']
        assert (cam.width, cam.height, cam.n_frames) == (32, 24, n) and cam.meta['encoding'] == 'png'
        frames = dict(cam.frames())
        assert all(np.array_equal(frames[k], s['imgs'][k]) for k in range(n))     # PNG: exact
        jp = dict(ep.cameras['observation/image'].frames([n - 1, 0]))
        assert set(jp) == {0, n - 1}
        assert np.abs(jp[0].astype(int) - s['smooth'][0]).mean() < 3            # JPEG: close, and RGB order
        with pytest.raises(DS.DatasetError, match='no frame'):
            list(cam.frames([n]))
    assert np.array_equal(ds.load(0).q, ds.load(0).q)                  # loading again gives the same episode
    with pytest.raises(DS.DatasetError, match='no episode'):
        ds.load(4)
    # the zlib / bytes encoded tensors decode too
    flat = R.parse_example(R.TFRecordFile(os.path.join(folder, 'toy_oxe-train.tfrecord-00000-of-00002')).read(0))
    steps, n, _ = ds._steps_from_example(flat)
    assert np.allclose(steps['observation/big'][0], order[0]['big'])
    assert np.array_equal(steps['observation/raw'][0], order[0]['raw'])
    # timed by a user-given fps
    ds10 = R.RldsDataset(folder, fps=10.0)
    ep = ds10.load(1)
    assert np.allclose(ep.t, np.arange(3) / 10.0) and ep.fps == 10.0 and 'fps' in ep.meta['time_source']
    assert ep.cameras['observation/image'].fps == 10.0


def test_example_without_features_json_and_timestamps(tmp):
    cv2 = pytest.importorskip('cv2')
    n = 6
    rng = np.random.default_rng(1)
    imgs = rng.integers(0, 256, (n, 10, 12, 3), dtype=np.uint8)
    pngs = [cv2.imencode('.png', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))[1].tobytes() for im in imgs]
    q = rng.normal(size=(n, 3)).astype(np.float32)
    ts = (100.0 + 0.05 * np.arange(n)).astype(np.float32)
    d = {'steps/is_first': TW.feature('int64', [1] + [0] * (n - 1)), 'steps/is_last': TW.feature('int64', [0] * n),
         'steps/observation/joint_pos': TW.feature('float', q.reshape(-1)),
         'steps/observation/timestamp': TW.feature('float', ts),
         'steps/observation/cam': TW.feature('bytes', pngs),
         'steps/language_instruction': TW.feature('bytes', [b'stack the cups'] * n),
         'steps/action': TW.feature('float', np.arange(2 * n, dtype=np.float32))}
    p = str(tmp / 'single.tfrecord')
    TW.write_tfrecord(p, [TW.example(d)])
    ds = DS.open_dataset(p)
    assert ds.format == 'rlds' and len(ds.episodes()) == 1
    ep = ds.load(0)
    assert np.array_equal(ep.q, q) and ep.joint_names == ['joint_pos_0', 'joint_pos_1', 'joint_pos_2']
    assert np.allclose(ep.t, ts.astype(np.float64) - float(ts[0])) and 'seconds' in ep.meta['time_source']
    assert ep.action.shape == (n, 2) and ep.task == 'stack the cups'
    assert any('is_last' in w for w in ep.meta['warnings'])           # never set: said, not hidden
    cam = ep.cameras['observation/cam']
    assert (cam.width, cam.height) == (12, 10) and np.array_equal(dict(cam.frames([2]))[2], imgs[2])


def test_sequence_example_records(tmp):
    cv2 = pytest.importorskip('cv2')
    n = 4
    img = np.full((8, 8, 3), 77, np.uint8)
    png = cv2.imencode('.png', img)[1].tobytes()
    lists = {'observation/state': [TW.feature('float', [i, 2 * i]) for i in range(n)],
             'observation/image': [TW.feature('bytes', [png]) for _ in range(n)],
             'action': [TW.feature('float', [0.5 * i]) for i in range(n)],
             'is_first': [TW.feature('int64', [int(i == 0)]) for i in range(n)],
             'is_last': [TW.feature('int64', [int(i == n - 1)]) for i in range(n)],
             'bad': [TW.feature('float', [1.0])]}
    ctx = {'episode_metadata/episode_id': TW.feature('bytes', [b'ep-42']),
           'language_instruction': TW.feature('bytes', [b'open the drawer'])}
    p = str(tmp / 'seq.tfrecord')
    TW.write_tfrecord(p, [TW.sequence_example(ctx, lists)] * 2)
    ds = R.RldsDataset(p)
    assert len(ds.episodes()) == 2
    ep = ds.load(1)
    assert ep.q.tolist() == [[i, 2 * i] for i in range(n)] and ep.action[:, 0].tolist() == [0, 0.5, 1.0, 1.5]
    assert ep.task == 'open the drawer' and ep.meta['episode_metadata']['episode_id'] == 'ep-42'
    assert any('bad' in w for w in ep.meta['warnings'])
    assert np.array_equal(dict(ep.cameras['observation/image'].frames())[3], img)


def test_damaged_episode_is_refused_plainly_and_sniffing(tmp):
    folder, _ = TW.write_oxe(str(tmp), 'dmg', {'train': [[3, 3]]})
    shard = os.path.join(folder, 'dmg-train.tfrecord-00000-of-00001')
    blob = bytearray(open(shard, 'rb').read())
    first_len = struct.unpack('<Q', blob[:8])[0]
    blob[12 + first_len // 2] ^= 0x55
    open(shard, 'wb').write(blob)
    ds = DS.open_dataset(folder)
    assert len(ds.episodes()) == 2
    with pytest.raises(DS.DatasetError) as e:
        ds.load(0)
    assert 'checksum' in str(e.value) and 'Traceback' not in str(e.value) and str(tmp) not in str(e.value)
    assert ds.load(1).q.shape == (3, 7)                                # the other episode still reads
    junk = tmp / 'junk.tfrecord'
    junk.write_bytes(b'not a tfrecord at all')
    assert not R.sniff(str(junk))
    other = tmp / 'plain'
    other.mkdir()
    (other / 'notes.txt').write_text('x')
    assert not R.sniff(str(other))
    with pytest.raises(DS.DatasetError):
        R.RldsDataset(str(other))


def test_truth_sidecar_is_returned(tmp):
    folder, _ = TW.write_oxe(str(tmp), 'withtruth', {'train': [[2]]})
    assert DS.open_dataset(folder).truth() is None
    doc = {'schema': 'dexvantage.truth/1', 'episodes': {}}
    with open(os.path.join(folder, 'dexvantage_truth.json'), 'w') as f:
        json.dump(doc, f)
    assert DS.open_dataset(folder).truth() == doc


def test_gzip_dataset(tmp):
    d, s = TW.make_episode(np.random.default_rng(2), 3, 8, 8)
    p = str(tmp / 'z-train.tfrecord-00000-of-00001')
    TW.write_tfrecord(p, [TW.example(d)], compress=True)
    with gzip.open(p) as f:
        assert len(f.read()) > 100
    ds = DS.open_dataset(str(tmp))
    assert ds.format == 'rlds' and np.array_equal(ds.load(0).q, s['state'])


# ---------------------------------------------------------------------------------------------------------------------
# the validation check

def test_tfrecord_crc_check_passes():
    from dexvantage.validation import references as VR
    from dexvantage.validation import suite
    suite.load_checks()
    spec = next(s for s in suite.CHECKS if s.id == 'formats.tfrecord_crc')
    assert spec.refs and all(k in VR.REFERENCES for k in spec.refs)
    ctx = suite.Ctx(None, None, None, suite.ValidationSettings(models=('formats',), quick=True))
    try:
        c = suite.run_one(spec, ctx)
    finally:
        ctx.close()
    assert c.passed, (c.error, c.detail)
    json.dumps(c.detail)
    assert c.detail['rfc3720_vectors'] == 5


def test_tfrecord_crc_check_catches_the_wrong_crc(monkeypatch):
    """CRC-32 (zlib's polynomial) in place of CRC-32C must fail the check."""
    import zlib

    from dexvantage.validation import suite
    suite.load_checks()
    monkeypatch.setattr(R, 'crc32c', lambda data, crc=0: zlib.crc32(bytes(data), crc))
    spec = next(s for s in suite.CHECKS if s.id == 'formats.tfrecord_crc')
    ctx = suite.Ctx(None, None, None, suite.ValidationSettings(models=('formats',), quick=True))
    try:
        c = suite.run_one(spec, ctx)
    finally:
        ctx.close()
    assert not c.passed and c.error >= 6
