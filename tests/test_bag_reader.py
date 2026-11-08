"""ROS 2 bags: the CDR codec, writer and readers of core/rosbag.py (ported from Vantage core/rosbag.py, with its tests
adapted: the CDR codec against hand-computed bytes and round trips of every type, the point-cloud and image helpers,
MCAP bags cross-checked with the mcap package both ways, rosbag2 SQLite bags, time ordering and damaged files), then
core/datasets/bag.py mapping robot-learning bags onto the shared Episode shape, checked against values written by hand
(SQLite bags through sqlite3 directly, MCAP files through the mcap package), and the formats.mcap_roundtrip check."""
import json
import os
import random
import shutil
import sqlite3
import string
import struct
import sys
import time

import numpy as np
import pytest

from dexvantage.core import rosbag as rb
from dexvantage.core.mathutil import rot_zyx

BRIEF_TYPES = [
    'builtin_interfaces/msg/Time', 'builtin_interfaces/msg/Duration', 'std_msgs/msg/Header', 'std_msgs/msg/String',
    'std_msgs/msg/ColorRGBA', 'geometry_msgs/msg/Vector3', 'geometry_msgs/msg/Point', 'geometry_msgs/msg/Quaternion',
    'geometry_msgs/msg/Pose', 'geometry_msgs/msg/Transform', 'geometry_msgs/msg/TransformStamped',
    'geometry_msgs/msg/Twist', 'geometry_msgs/msg/PoseWithCovariance', 'geometry_msgs/msg/TwistWithCovariance',
    'tf2_msgs/msg/TFMessage', 'nav_msgs/msg/Odometry', 'sensor_msgs/msg/Image', 'sensor_msgs/msg/CompressedImage',
    'sensor_msgs/msg/CameraInfo', 'sensor_msgs/msg/RegionOfInterest', 'sensor_msgs/msg/PointCloud2',
    'sensor_msgs/msg/PointField', 'sensor_msgs/msg/Imu', 'sensor_msgs/msg/NavSatFix', 'sensor_msgs/msg/NavSatStatus',
    'sensor_msgs/msg/TimeReference', 'visualization_msgs/msg/Marker', 'visualization_msgs/msg/MarkerArray',
    'sensor_msgs/msg/JointState', 'std_msgs/msg/Float64MultiArray', 'std_msgs/msg/Float32MultiArray']

_INT = {'int8': (-128, 127), 'uint8': (0, 255), 'byte': (0, 255), 'char': (0, 255), 'int16': (-32768, 32767),
        'uint16': (0, 65535), 'int32': (-2**31, 2**31 - 1), 'uint32': (0, 2**32 - 1),
        'int64': (-2**63, 2**63 - 1), 'uint64': (0, 2**64 - 1)}


def _prim(t, rng):
    if t == 'bool':
        return rng.random() < 0.5
    if t in _INT:
        return rng.randint(*_INT[t])
    if t == 'float32':
        return float(np.float32(rng.uniform(-1e3, 1e3)))       # exactly representable
    if t == 'float64':
        return rng.uniform(-1e6, 1e6)
    return ''.join(rng.choice(string.ascii_letters + 'é→') for _ in range(rng.randint(0, 9)))


def _random_msg(t, rng, depth=0):
    """A message of type t with every field set to a random value (sequences 0..3 long)."""
    d = {}
    for f in rb._TYPES.spec(t).fields:
        simple = f.type in rb._PRIM or f.type == 'string'
        if f.kind == 0:
            d[f.name] = _prim(f.type, rng) if simple else _random_msg(f.type, rng, depth + 1)
            continue
        n = f.n if f.kind == 1 else rng.randint(0, 3 if depth < 2 else 1)
        if f.type in rb._BYTES:
            d[f.name] = bytes(rng.randint(0, 255) for _ in range(n if f.kind == 1 else rng.randint(0, 13)))
        elif simple:
            d[f.name] = [_prim(f.type, rng) for _ in range(n)]
        else:
            d[f.name] = [_random_msg(f.type, rng, depth + 1) for _ in range(n)]
    return d


@pytest.fixture(autouse=True)
def _restore_definitions():
    """Types a test register()s are removed afterwards, so other test modules see the stock MSG_DEFS."""
    before = set(rb.MSG_DEFS)
    yield
    for t in set(rb.MSG_DEFS) - before:
        del rb.MSG_DEFS[t]
        rb._TYPES._specs.pop(t, None)
        for cache in (rb._TYPES._enc, rb._TYPES._dec):
            for key in [k for k in cache if k[0] == t]:
                del cache[key]


@pytest.fixture
def tmp_bags(tmp_path):
    """A folder for bags, deleted afterwards (/tmp is often RAM)."""
    yield tmp_path
    shutil.rmtree(tmp_path, ignore_errors=True)


def _compressions():
    out = ['none']
    for name, mod in (('zstd', 'zstandard'), ('lz4', 'lz4.frame')):
        try:
            __import__(mod)
            out.append(name)
        except ImportError:
            pass
    return out


# ---------------------------------------------------------------------------------------------------------------------
# CDR

def test_header_bytes_known():
    data = rb.encode('std_msgs/msg/Header', rb.header(1_500_000_000, 'abc'))
    assert data.hex(' ') == '00 01 00 00 01 00 00 00 00 65 cd 1d 04 00 00 00 61 62 63 00'
    assert rb.decode('std_msgs/msg/Header', data) == {'stamp': {'sec': 1, 'nanosec': 500_000_000},
                                                      'frame_id': 'abc'}


def test_alignment_hand_computed():
    """A uint8 before a float64, strings of every length mod 4 before 8- and 2-byte fields, against a layout
    computed by hand with struct (offsets relative to the payload after the 4-byte header)."""
    rb.register('dexvantage_test_msgs/msg/Align', 'uint8 a\nfloat64 b\nstring s\nfloat64 c\nstring u\nint16 d\n')

    def by_hand(a, b, s, c, u, d):
        out = bytearray(struct.pack('<B', a))
        out += bytes(7) + struct.pack('<d', b)                   # b at 8
        sb = s.encode() + b'\0'
        out += struct.pack('<I', len(sb)) + sb                   # 16: length, then the bytes
        out += bytes(-len(out) % 8) + struct.pack('<d', c)
        ub = u.encode() + b'\0'
        out += bytes(-len(out) % 4) + struct.pack('<I', len(ub)) + ub
        out += bytes(-len(out) % 2) + struct.pack('<h', d)
        return b'\x00\x01\x00\x00' + bytes(out)

    for n in range(9):
        for m in range(5):
            msg = {'a': 200, 'b': -1.25, 's': 'x' * n, 'c': 3.5e10, 'u': 'y' * m, 'd': -7}
            data = rb.encode('dexvantage_test_msgs/msg/Align', msg)
            assert data == by_hand(**msg), (n, m)
            assert rb.decode('dexvantage_test_msgs/msg/Align', data) == msg


def test_empty_sequence_elements_not_aligned():
    rb.register('dexvantage_test_msgs/msg/Seq', 'uint8 a\nfloat64[] d\nuint8 b\n')
    empty = rb.encode('dexvantage_test_msgs/msg/Seq', {'a': 1, 'd': [], 'b': 2})
    assert empty[4:].hex(' ') == '01 00 00 00 00 00 00 00 02'               # no padding to 8 for no elements
    one = rb.encode('dexvantage_test_msgs/msg/Seq', {'a': 1, 'd': [2.0], 'b': 3})
    assert one[4:] == bytes([1, 0, 0, 0, 1, 0, 0, 0]) + struct.pack('<d', 2.0) + b'\x03'
    assert rb.decode('dexvantage_test_msgs/msg/Seq', one) == {'a': 1, 'd': [2.0], 'b': 3}


@pytest.mark.parametrize('t', BRIEF_TYPES)
def test_round_trip_every_type(t):
    rng = random.Random(hash(t) & 0xffff)
    for _ in range(20):
        msg = _random_msg(t, rng)
        data = rb.encode(t, msg)
        assert data[:4] == b'\x00\x01\x00\x00'
        back = rb.decode(t, data)
        assert back == msg
        assert rb.encode(t, back) == data
    # defaults: an empty dict encodes every field at its default, and that decodes and re-encodes identically
    zero = rb.encode(t, {})
    assert rb.encode(t, rb.decode(t, zero)) == zero


def test_defaults_and_errors():
    q = rb.decode('geometry_msgs/msg/Pose', rb.encode('geometry_msgs/msg/Pose', {}))
    assert q['orientation'] == {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
    assert rb.constants('visualization_msgs/msg/Marker')['CUBE'] == 1
    assert rb.constants('sensor_msgs/msg/NavSatStatus')['STATUS_NO_FIX'] == -1
    with pytest.raises(ValueError, match='no field'):
        rb.encode('geometry_msgs/msg/Point', {'x': 1.0, 'w': 2.0})
    with pytest.raises(ValueError):
        rb.encode('sensor_msgs/msg/Imu', {'orientation_covariance': [0.0] * 8})
    with pytest.raises(ValueError):
        rb.encode('std_msgs/msg/Header', {'stamp': {'sec': 2**31}})              # int32 overflow
    with pytest.raises(ValueError, match='unknown message type'):
        rb.encode('nope_msgs/msg/Nothing', {})
    good = rb.encode('sensor_msgs/msg/NavSatFix', {'latitude': 12.5, 'header': rb.header(3, 'gnss')})
    for cut in (3, 10, 20, len(good) - 1):
        with pytest.raises(ValueError):
            rb.decode('sensor_msgs/msg/NavSatFix', good[:cut])
    with pytest.raises(ValueError, match='XCDR1'):
        rb.decode('std_msgs/msg/String', b'\x00\x07\x00\x00' + good[4:])
    # short type names and numpy inputs
    k = np.array([[800.0, 0, 640], [0, 800, 360], [0, 0, 1]])
    info = rb.camera_info(1280, 720, 'plumb_bob', np.array([0.1, -0.05, 0, 0, 0.01]), k, t_ns=5, frame_id='cam')
    back = rb.decode('sensor_msgs/CameraInfo', rb.encode('sensor_msgs/CameraInfo', dict(info, k=k)))
    assert back['k'] == k.reshape(9).tolist() and back['p'][:3] == [800.0, 0.0, 640.0] and back['r'][4] == 1.0
    assert back['d'] == [0.1, -0.05, 0.0, 0.0, 0.01]


def test_big_endian_decoded():
    msg = {'header': rb.header(7_000_000_123, 'gps'), 'status': {'status': -1, 'service': 9},
           'latitude': 48.1, 'longitude': 11.5, 'altitude': 520.0, 'position_covariance': [float(i) for i in range(9)],
           'position_covariance_type': 2}
    le = rb.encode('sensor_msgs/msg/NavSatFix', msg)[4:]
    # the same message written big-endian by hand: every primitive byte-swapped at the same offsets
    hb = struct.pack('>iI', 7, 123) + struct.pack('>I', 4) + b'gps\0'
    hb += struct.pack('>bxH', -1, 9)
    hb += bytes(-len(hb) % 8)                                   # latitude aligns to 8
    hb += struct.pack('>3d', 48.1, 11.5, 520.0) + struct.pack('>9d', *range(9)) + b'\x02'
    assert len(hb) == len(le)
    assert rb.decode('sensor_msgs/msg/NavSatFix', b'\x00\x00\x00\x00' + hb) == msg


def test_schema_text_like_rosbag2():
    text = rb.schema_text('nav_msgs/msg/Odometry')
    assert text.startswith(rb.MSG_DEFS['nav_msgs/msg/Odometry'])
    names = [ln[5:] for ln in text.splitlines() if ln.startswith('MSG: ')]
    assert names == ['geometry_msgs/PoseWithCovariance', 'geometry_msgs/Pose', 'geometry_msgs/Point',
                     'geometry_msgs/Quaternion', 'geometry_msgs/TwistWithCovariance', 'geometry_msgs/Twist',
                     'geometry_msgs/Vector3', 'std_msgs/Header', 'builtin_interfaces/Time']
    assert all(ln == '=' * 80 for ln in text.splitlines() if ln.startswith('='))
    assert '\n\n' + '=' * 80 + '\nMSG: geometry_msgs/PoseWithCovariance\n' in text


# ---------------------------------------------------------------------------------------------------------------------
# point clouds and images

def test_pointcloud2_mixed_fields_padding():
    dt = np.dtype({'names': ['x', 'y', 'z', 'intensity', 'ring', 't', 'flag', 'rgb'],
                   'formats': ['<f4', '<f4', '<f4', '<u2', '<u1', '<f8', '<i1', ('<u1', (3,))],
                   'offsets': [0, 4, 8, 12, 14, 16, 24, 25], 'itemsize': 32})
    rng = np.random.default_rng(3)
    pts = np.zeros(257, dt)
    for n in ('x', 'y', 'z'):
        pts[n] = rng.normal(0, 30, 257)
    pts['intensity'] = rng.integers(0, 65535, 257)
    pts['ring'] = rng.integers(0, 128, 257)
    pts['t'] = rng.uniform(0, 0.1, 257)
    pts['flag'] = rng.integers(-128, 127, 257)
    pts['rgb'] = rng.integers(0, 255, (257, 3))
    msg = rb.pointcloud2(pts, 42, 'lidar')
    assert msg['point_step'] == 32 and msg['row_step'] == 32 * 257 and msg['height'] == 1 and msg['is_dense']
    assert [(f['name'], f['offset'], f['datatype'], f['count']) for f in msg['fields']] == [
        ('x', 0, 7, 1), ('y', 4, 7, 1), ('z', 8, 7, 1), ('intensity', 12, 4, 1), ('ring', 14, 2, 1),
        ('t', 16, 8, 1), ('flag', 24, 1, 1), ('rgb', 25, 2, 3)]
    # independent check of the data layout: point 5's z and t read straight from the bytes
    assert struct.unpack_from('<f', msg['data'], 5 * 32 + 8)[0] == pts['z'][5]
    assert struct.unpack_from('<d', msg['data'], 5 * 32 + 16)[0] == pts['t'][5]
    back = rb.pointcloud2_array(rb.decode('sensor_msgs/msg/PointCloud2', rb.encode('sensor_msgs/msg/PointCloud2', msg)))
    assert back.shape == (257,) and back.dtype.itemsize == 32
    for n in dt.names:
        assert np.array_equal(back[n], pts[n]), n
    pts['x'][3] = np.nan
    assert not rb.pointcloud2(pts, 0, 'lidar')['is_dense']


def test_pointcloud2_organised_bigendian_rowpad_unknown_field():
    h, w = 3, 4
    be = np.dtype({'names': ['x', 'y', 'z', 'i'], 'formats': ['>f4', '>f4', '>f4', '>u2'],
                   'offsets': [0, 4, 8, 12], 'itemsize': 16})
    pts = np.zeros((h, w), be)
    pts['x'] = np.arange(12).reshape(h, w) * 1.5
    pts['y'] = -pts['x']
    pts['z'] = 0.25
    pts['i'] = np.arange(12).reshape(h, w) * 1000
    row_step = w * 16 + 8                                            # 8 padding bytes after every row
    data = b''.join(pts[r].tobytes() + b'\xee' * 8 for r in range(h))
    fields = [{'name': n, 'offset': o, 'datatype': d, 'count': 1}
              for n, o, d in (('x', 0, 7), ('y', 4, 7), ('z', 8, 7), ('i', 12, 4),
                              ('odd', 14, 42))]                     # an unknown datatype: skipped
    msg = {'header': rb.header(0, 'l'), 'height': h, 'width': w, 'fields': fields, 'is_bigendian': True,
           'point_step': 16, 'row_step': row_step, 'data': data, 'is_dense': True}
    back = rb.pointcloud2_array(rb.decode('sensor_msgs/msg/PointCloud2', rb.encode('sensor_msgs/msg/PointCloud2', msg)))
    assert back.shape == (h, w) and back.dtype.names == ('x', 'y', 'z', 'i')
    for n in back.dtype.names:
        assert np.array_equal(back[n], pts[n].astype(pts[n].dtype.newbyteorder('='))), n
    # written back it is little-endian and dense rows, same values
    again = rb.pointcloud2_array(rb.pointcloud2(back, 0, 'l'))
    assert again.shape == (h, w) and np.array_equal(again['i'], back['i'])
    with pytest.raises(ValueError):
        rb.pointcloud2(np.zeros(3, [('x', '<u8')]), 0, 'l')


def test_images_round_trip():
    rng = np.random.default_rng(5)
    rgb = rng.integers(0, 256, (37, 53, 3), dtype=np.uint8)
    for img, enc in ((rgb, 'rgb8'), (rgb[:, :, 0], 'mono8'),
                     (rng.integers(0, 65536, (9, 11), dtype=np.uint16), 'mono16'),
                     (rng.normal(size=(5, 7)).astype(np.float32), '32FC1'),
                     (rng.integers(0, 256, (6, 8, 4), dtype=np.uint8), 'rgba8')):
        msg = rb.image_msg(img, 10, 'cam')
        channels = img.shape[2] if img.ndim == 3 else 1
        assert msg['encoding'] == enc and msg['step'] == img.shape[1] * img.itemsize * channels
        back = rb.image_array(rb.decode('sensor_msgs/msg/Image', rb.encode('sensor_msgs/msg/Image', msg)))
        assert back.dtype == img.dtype and np.array_equal(back, img), enc
    bgr = rb.image_msg(rgb[:, :, ::-1], 0, 'cam', 'bgr8')
    assert np.array_equal(rb.image_array(bgr)[:, :, ::-1], rgb)
    assert np.array_equal(rb.image_array(bgr, to_rgb=True), rgb)
    bayer = rb.image_msg(rgb[:, :, 1], 0, 'cam', 'bayer_rggb8')
    assert np.array_equal(rb.image_array(bayer), rgb[:, :, 1])
    # step padding and big-endian 16-bit
    m16 = rng.integers(0, 65536, (4, 5), dtype=np.uint16)
    rows = b''.join(m16[r].astype('>u2').tobytes() + b'\0\0\0\0\0\0' for r in range(4))
    msg = {'header': rb.header(0, 'c'), 'height': 4, 'width': 5, 'encoding': 'mono16', 'is_bigendian': 1,
           'step': 16, 'data': rows}
    assert np.array_equal(rb.image_array(msg), m16)
    with pytest.raises(ValueError):
        rb.image_msg(rgb, 0, 'cam', 'mono16')
    with pytest.raises(ValueError):
        rb.image_array(dict(msg, encoding='nv12_odd'))


def test_compressed_images():
    cv2 = pytest.importorskip('cv2', reason='opencv-python (a core dependency) does not import here')
    rng = np.random.default_rng(6)
    rgb = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
    ok, png = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    msg = rb.compressed_image(png.tobytes(), 5, 'cam', 'png')
    back = rb.decode('sensor_msgs/msg/CompressedImage', rb.encode('sensor_msgs/msg/CompressedImage', msg))
    assert back['format'] == 'png' and np.array_equal(rb.image_array(back), rgb)
    smooth = np.repeat(np.linspace(0, 255, 64).astype(np.uint8)[None, :, None], 48, 0).repeat(3, 2)
    smooth[..., 0] = 200
    ok, jpg = cv2.imencode('.jpg', cv2.cvtColor(smooth, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    dec = rb.image_array(rb.compressed_image(jpg.tobytes(), 0, 'cam'))
    assert dec.shape == smooth.shape and np.abs(dec.astype(int) - smooth).mean() < 3
    assert abs(int(dec[..., 0].mean()) - 200) < 3                       # red stays red: RGB order
    d16 = rng.integers(0, 65535, (8, 9), dtype=np.uint16)
    ok, p16 = cv2.imencode('.png', d16)
    assert np.array_equal(rb.image_array(rb.compressed_image(p16.tobytes(), 0, 'depth', 'png')), d16)
    with pytest.raises(ValueError):
        rb.image_array(rb.compressed_image(b'not an image', 0, 'cam'))


def test_helpers():
    for t in (0, 1, 999_999_999, 1_000_000_000, 1_700_000_000_123_456_789, -1, -1_500_000_000):
        st = rb.stamp(t)
        assert 0 <= st['nanosec'] < 1_000_000_000 and rb.stamp_ns(st) == t
    R = rot_zyx(30.0, -10.0, 5.0)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, [1.2, -0.4, 1.9]
    ts = rb.transform_stamped(T, 77, 'base_link', 'wrist_cam')
    assert ts['header']['frame_id'] == 'base_link' and ts['child_frame_id'] == 'wrist_cam'
    back = rb.decode('geometry_msgs/msg/TransformStamped', rb.encode('geometry_msgs/msg/TransformStamped', ts))
    assert np.allclose(rb.transform_matrix(back), T, atol=1e-12)
    q = rb.quat_msg(R)
    assert abs(q['x'] ** 2 + q['y'] ** 2 + q['z'] ** 2 + q['w'] ** 2 - 1) < 1e-12
    # independent: rotating x by the quaternion (q v q*) equals R @ x
    w, v = q['w'], np.array([q['x'], q['y'], q['z']])
    x = np.array([1.0, 0, 0])
    assert np.allclose(x + 2 * w * np.cross(v, x) + 2 * np.cross(v, np.cross(v, x)), R @ x)
    mk = {'header': rb.header(9, 'world'), 'ns': 'gt', 'id': 3, 'type': 1, 'action': 0,
          'pose': {'position': {'x': 0.5, 'y': 0.1, 'z': 0.08}, 'orientation': rb.quat_msg(R)},
          'scale': {'x': 0.045, 'y': 0.018, 'z': 0.015}, 'color': {'r': 1.0, 'g': 0.6, 'b': 0.0, 'a': 0.5}}
    m2 = rb.decode('visualization_msgs/msg/Marker', rb.encode('visualization_msgs/msg/Marker', mk))
    assert m2['type'] == rb.constants('visualization_msgs/msg/Marker')['CUBE'] and m2['scale']['x'] == 0.045
    arr = rb.decode('visualization_msgs/msg/MarkerArray', rb.encode('visualization_msgs/msg/MarkerArray',
                                                                    {'markers': [mk, mk]}))
    assert len(arr['markers']) == 2 and arr['markers'][1]['id'] == 3


# ---------------------------------------------------------------------------------------------------------------------
# bags

def _check_with_mcap(path, sent_counts, sent_data):
    """Read a .mcap we wrote with the mcap package (CRCs validated): schemas, channels, counts and bytes."""
    from mcap.reader import make_reader
    from mcap.stream_reader import StreamReader
    with open(path, 'rb') as f:
        rd = make_reader(f, validate_crcs=True)
        summary = rd.get_summary()
        assert summary.statistics.message_count == sum(sent_counts.values())
        assert rd.get_header().profile == 'ros2' and rd.get_header().library == 'dexvantage'
        got_counts, got = {}, {}
        for schema, channel, message in rd.iter_messages(log_time_order=True):
            assert schema.encoding == 'ros2msg' and channel.message_encoding == 'cdr'
            assert schema.data.decode() == rb.schema_text(schema.name)
            assert 'offered_qos_profiles' in channel.metadata
            got_counts[channel.topic] = got_counts.get(channel.topic, 0) + 1
            got.setdefault(channel.topic, []).append((message.log_time, message.data))
        assert got_counts == sent_counts
        for topic, lst in sent_data.items():
            assert sorted(got[topic]) == sorted(lst), topic
        chunks = summary.chunk_indexes
    with open(path, 'rb') as f:                     # a linear read of every record checks the data-section CRC
        recs = list(StreamReader(f, validate_crcs=True).records)
    assert type(recs[-1]).__name__ == 'Footer'
    return chunks


@pytest.mark.parametrize('comp', _compressions())
def test_bag_three_topics_chunked(tmp_bags, comp):
    pytest.importorskip('mcap', reason='mcap (the test extra, an independent MCAP reader) is not installed')
    rng = np.random.default_rng(11)
    big = rng.integers(0, 256, (512, 683, 3), dtype=np.uint8)            # ~1 MB rgb8
    small = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    path = str(tmp_bags / f'episode_{comp}')
    sent = {'/imu': [], '/tf': [], '/cam': []}
    types = {'/imu': 'sensor_msgs/msg/Imu', '/tf': 'tf2_msgs/msg/TFMessage', '/cam': 'sensor_msgs/msg/Image'}
    with rb.BagWriter(path, compression=comp, chunk_size=1 << 20) as bag:
        bag.add_topic('/imu', 'sensor_msgs/msg/Imu')
        bag.add_topic('/tf', 'tf2_msgs/msg/TFMessage')
        bag.add_topic('/cam', 'sensor_msgs/msg/Image')
        for i in range(1000):
            t = 1_700_000_000_000_000_000 + i * 10_000_000
            imu = {'header': rb.header(t, 'imu'), 'angular_velocity': {'z': i * 1e-3},
                   'linear_acceleration': {'x': 0.1, 'z': 9.81}}
            tf = {'transforms': [rb.transform_stamped(np.eye(4), t, 'odom', 'base_link')]}
            cam = rb.image_msg(big if i % 50 == 0 else small, t + 3, 'wrist_optical')
            for topic, m, tt in (('/imu', imu, t + 1), ('/tf', tf, t + 2), ('/cam', cam, t + 3)):
                data = rb.encode(types[topic], m)
                bag.write(topic, tt, m if topic != '/tf' else data)          # dicts and pre-encoded bytes
                sent[topic].append((tt, data))
        res = bag.close()
    assert res['messages'] == 3000 and res['topics'] == {'/imu': 1000, '/tf': 1000, '/cam': 1000}
    assert sorted(os.listdir(path)) == sorted(['metadata.yaml', f'episode_{comp}_0.mcap'])
    chunks = _check_with_mcap(res['file'], res['topics'], sent)
    assert len(chunks) >= 20 and {c.compression for c in chunks} == {'' if comp == 'none' else comp}
    with rb.BagReader(path) as r:
        assert r.topics == {'/imu': 'sensor_msgs/msg/Imu', '/tf': 'tf2_msgs/msg/TFMessage',
                            '/cam': 'sensor_msgs/msg/Image'}
        assert r.counts == res['topics'] and r.message_count == 3000 and not r.warnings
        assert r.start_ns == 1_700_000_000_000_000_001 and r.end_ns == 1_700_000_000_000_000_000 + 999 * 10_000_000 + 3
        prev, n = -1, 0
        for topic, t, msg in r.messages():
            assert t >= prev
            prev = t
            if topic == '/cam' and n % 150 == 2:
                img = rb.image_array(msg)
                assert np.array_equal(img, big if img.shape[0] == 512 else small)
            n += 1
        assert n == 3000
        raw = [(t, m) for _, t, m in r.messages('/tf', decode=False)]
        assert raw == sent['/tf']
        window = list(r.messages(['/imu', '/cam'], start_ns=r.start_ns + 100_000_000, end_ns=r.start_ns + 200_000_000))
        assert len(window) == 20 and {w[0] for w in window} == {'/imu', '/cam'}
        assert all(r.start_ns + 100_000_000 <= w[1] < r.start_ns + 200_000_000 for w in window)
    import yaml
    meta = yaml.safe_load(open(os.path.join(path, 'metadata.yaml')))['rosbag2_bagfile_information']
    assert meta['version'] == 5 and meta['storage_identifier'] == 'mcap' and meta['message_count'] == 3000
    assert meta['relative_file_paths'] == [f'episode_{comp}_0.mcap']
    assert meta['starting_time']['nanoseconds_since_epoch'] == r.start_ns
    assert meta['duration']['nanoseconds'] == r.end_ns - r.start_ns
    assert {x['topic_metadata']['name']: x['message_count'] for x in meta['topics_with_message_count']} == res['topics']


def test_out_of_order_writes_come_back_in_time_order(tmp_bags):
    rng = random.Random(4)
    path = str(tmp_bags / 'shuffled')
    times = [rng.randint(0, 10**12) for _ in range(3000)]
    with rb.BagWriter(path, compression='none', chunk_size=4096) as bag:   # many chunks with overlapping times
        bag.add_topic('/a', 'std_msgs/msg/String')
        bag.add_topic('/b', 'geometry_msgs/msg/Point')
        for i, t in enumerate(times):
            if i % 2:
                bag.write('/a', t, {'data': str(t)})
            else:
                bag.write('/b', t, {'x': float(t)})
    with rb.BagReader(path) as r:
        got = list(r.messages())
        assert [g[1] for g in got] == sorted(times)
        assert all((m['data'] == str(t)) if tp == '/a' else (m['x'] == float(t)) for tp, t, m in got)
        assert [g[1] for g in r.messages('/b')] == sorted(t for i, t in enumerate(times) if i % 2 == 0)
        lo, hi = sorted(times)[1000], sorted(times)[2000]
        assert [g[1] for g in r.messages(start_ns=lo, end_ns=hi)] == [t for t in sorted(times) if lo <= t < hi]
        assert r.start_ns == min(times) and r.end_ns == max(times)


@pytest.mark.parametrize('variant', ['zstd', 'lz4', 'none', 'unchunked', 'no_stats', 'no_index', 'chunk_index_only'])
def test_files_written_by_mcap_package(tmp_bags, variant):
    """Files from mcap.writer.Writer (schemas and channels inside chunks, other index layouts) read by BagReader."""
    pytest.importorskip('mcap', reason='mcap (the test extra, an independent MCAP reader) is not installed')
    from mcap.writer import CompressionType, IndexType, Writer
    if variant == 'zstd':
        pytest.importorskip('zstandard', reason='zstandard (the bags extra) is not installed')
    if variant == 'lz4':
        pytest.importorskip('lz4', reason='lz4 (the bags extra) is not installed')
    kw = {'compression': {'zstd': CompressionType.ZSTD, 'lz4': CompressionType.LZ4}.get(variant, CompressionType.NONE),
          'chunk_size': 2048}
    if variant == 'unchunked':
        kw['use_chunking'] = False
    if variant == 'no_stats':
        kw['use_statistics'] = False
    if variant == 'no_index':
        kw['index_types'] = IndexType.NONE
    if variant == 'chunk_index_only':
        kw.update(index_types=IndexType.CHUNK, use_statistics=False)
    path = str(tmp_bags / 'theirs.mcap')
    rng = random.Random(9)
    sent = []
    with open(path, 'wb') as f:
        w = Writer(f, **kw)
        w.start(profile='ros2', library='test')
        chans = {}
        for t, topic in (('sensor_msgs/msg/NavSatFix', '/gnss'), ('nav_msgs/msg/Odometry', '/odom'),
                         ('visualization_msgs/msg/MarkerArray', '/boxes')):
            sid = w.register_schema(name=t, encoding='ros2msg', data=rb.schema_text(t).encode())
            chans[topic] = (w.register_channel(topic=topic, message_encoding='cdr', schema_id=sid), t)
        for i in range(400):
            topic = rng.choice(list(chans))
            msg = _random_msg(chans[topic][1], rng)
            t = rng.randint(0, 10**10)
            w.add_message(chans[topic][0], log_time=t, data=rb.encode(chans[topic][1], msg), publish_time=t)
            sent.append((t, topic, msg))
        w.finish()
    with rb.BagReader(path) as r:
        assert r.topics == {'/gnss': 'sensor_msgs/msg/NavSatFix', '/odom': 'nav_msgs/msg/Odometry',
                            '/boxes': 'visualization_msgs/msg/MarkerArray'}
        assert r.message_count == 400 and sum(r.counts.values()) == 400
        assert r.start_ns == min(s[0] for s in sent) and r.end_ns == max(s[0] for s in sent)
        got = list(r.messages())
        assert [g[1] for g in got] == sorted(s[0] for s in sent)
        assert sorted((g[1], g[0], repr(g[2])) for g in got) == sorted((s[0], s[1], repr(s[2])) for s in sent)
        assert not r.decode_errors
        assert bool(r.warnings) == (variant in ('unchunked', 'no_index'))       # those are read by a linear scan


def test_bag_own_definitions_and_undecodable(tmp_bags):
    """A type MSG_DEFS does not know is decoded from the bag's schema; a bag's own definition of a known name wins;
    a non-CDR channel and a corrupt message come back as bytes without stopping the iteration."""
    pytest.importorskip('mcap', reason='mcap (the test extra, an independent MCAP reader) is not installed')
    from mcap.writer import CompressionType, Writer
    path = str(tmp_bags / 'vendor.mcap')
    vendor = ('std_msgs/Header header\nuint8 MODE_A=1\nuint8 mode\nfloat64[] ranges\nstring[] names\n'
              '\n' + '=' * 80 + '\nMSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n'
              '\n' + '=' * 80 + '\nMSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n')
    with open(path, 'wb') as f:
        w = Writer(f, compression=CompressionType.NONE)
        w.start(profile='ros2', library='test')
        s1 = w.register_schema('acme_msgs/msg/Scan', 'ros2msg', vendor.encode())
        s2 = w.register_schema('std_msgs/msg/String', 'ros2msg', b'int32 data\n')        # an odd String
        s3 = w.register_schema('foo', 'jsonschema', b'{}')
        c1 = w.register_channel('/scan', 'cdr', s1)
        c2 = w.register_channel('/odd', 'cdr', s2)
        c3 = w.register_channel('/json', 'json', s3)
        c4 = w.register_channel('/broken', 'cdr', w.register_schema('sensor_msgs/msg/Imu', 'ros2msg',
                                                                    rb.schema_text('sensor_msgs/msg/Imu').encode()))
        body = (struct.pack('<iI', 3, 4) + struct.pack('<I', 2) + b'l\0' + b'\x01' + bytes(1)   # mode at 14, pad
                + struct.pack('<I', 2) + bytes(4) + struct.pack('<2d', 1.5, 2.5)            # ranges at 24
                + struct.pack('<I', 1) + struct.pack('<I', 3) + b'ab\0')
        w.add_message(c1, log_time=10, data=b'\x00\x01\x00\x00' + body, publish_time=10)
        w.add_message(c2, log_time=20, data=b'\x00\x01\x00\x00' + struct.pack('<i', -5), publish_time=20)
        w.add_message(c3, log_time=30, data=b'{"a": 1}', publish_time=30)
        w.add_message(c4, log_time=40, data=b'\x00\x01\x00\x00\x01\x02', publish_time=40)     # truncated Imu
        w.add_message(c2, log_time=50, data=b'\x00\x01\x00\x00' + struct.pack('<i', 6), publish_time=50)
        w.finish()
    with rb.BagReader(path) as r:
        got = list(r.messages())
    assert [g[0] for g in got] == ['/scan', '/odd', '/json', '/broken', '/odd']
    assert got[0][2] == {'header': {'stamp': {'sec': 3, 'nanosec': 4}, 'frame_id': 'l'}, 'mode': 1,
                         'ranges': [1.5, 2.5], 'names': ['ab']}
    assert got[1][2] == {'data': -5} and got[4][2] == {'data': 6}
    assert got[2][2] == b'{"a": 1}' and isinstance(got[3][2], bytes)
    assert r.decode_errors == {'/broken': 1}
    with rb.BagReader(path) as r:                     # lazy decoding of raw messages with the bag's definitions
        raw = {tp: m for tp, _, m in r.messages(decode=False)}
        assert r.decode('/scan', raw['/scan']) == got[0][2] and r.decode('/odd', raw['/odd']) == {'data': 6}
        for tp in ('/json', '/broken', '/nothing'):
            with pytest.raises(ValueError):
                r.decode(tp, raw.get(tp, b''))
    assert rb.decode('std_msgs/msg/String', rb.encode('std_msgs/msg/String', {'data': 'ok'})) == {'data': 'ok'}


def _make_db3(path, rows, topics, definitions=None):
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT NOT NULL, type TEXT NOT NULL, '
                'serialization_format TEXT NOT NULL, offered_qos_profiles TEXT NOT NULL)')
    con.execute('CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER NOT NULL, timestamp INTEGER NOT NULL, '
                'data BLOB NOT NULL)')
    con.execute('CREATE INDEX timestamp_idx ON messages (timestamp ASC)')
    for tid, (name, typ) in topics.items():
        con.execute('INSERT INTO topics VALUES (?, ?, ?, ?, ?)', (tid, name, typ, 'cdr', ''))
    if definitions:
        con.execute('CREATE TABLE message_definitions(id INTEGER PRIMARY KEY, topic_type TEXT NOT NULL, '
                    'encoding TEXT NOT NULL, encoded_message_definition TEXT NOT NULL, type_description_hash TEXT)')
        for typ, text in definitions.items():
            con.execute('INSERT INTO message_definitions(topic_type, encoding, encoded_message_definition, '
                        'type_description_hash) VALUES (?, ?, ?, ?)', (typ, 'ros2msg', text, ''))
    con.executemany('INSERT INTO messages(topic_id, timestamp, data) VALUES (?, ?, ?)', rows)
    con.commit()
    con.close()


def test_db3_bags(tmp_bags):
    rng = random.Random(2)
    d = tmp_bags / 'sqlite_bag'
    d.mkdir()
    topics = {1: ('/imu', 'sensor_msgs/msg/Imu'), 2: ('/fix', 'sensor_msgs/msg/NavSatFix'),
              3: ('/custom', 'acme_msgs/msg/Speed'), 4: ('/unknown', 'acme_msgs/msg/Mystery')}
    rows, sent = [], []
    for i in range(600):
        tid = rng.choice([1, 2, 3, 4])
        t = rng.randint(0, 10**9)
        if tid in (1, 2):
            msg = _random_msg(topics[tid][1], rng)
            data = rb.encode(topics[tid][1], msg)
        elif tid == 3:
            msg = {'kmh': float(i), 'valid': True}
            data = b'\x00\x01\x00\x00' + struct.pack('<d?', float(i), True)
        else:
            data = msg = b'\x00\x01\x00\x00\x09'
        rows.append((tid, t, data))
        sent.append((t, topics[tid][0], msg))
    _make_db3(str(d / 'sqlite_bag_0.db3'), rows, topics, {'acme_msgs/msg/Speed': 'float64 kmh\nbool valid\n'})
    for target in (str(d / 'sqlite_bag_0.db3'), str(d)):
        with rb.BagReader(target) as r:
            assert r.storage == 'sqlite3'
            assert r.topics == {v[0]: v[1] for v in topics.values()}
            assert r.message_count == 600 and r.start_ns == min(s[0] for s in sent)
            got = list(r.messages())
            assert [g[1] for g in got] == sorted(s[0] for s in sent)
            assert sorted((g[1], g[0], repr(g[2])) for g in got) == sorted((s[0], s[1], repr(s[2])) for s in sent)
            only = list(r.messages(['/fix'], start_ns=200_000_000, end_ns=600_000_000))
            assert [g[1] for g in only] == sorted(s[0] for s in sent if s[1] == '/fix' and 2e8 <= s[0] < 6e8)
    # with a rosbag2 metadata.yaml naming the file
    import yaml
    meta = {'rosbag2_bagfile_information': {'version': 5, 'storage_identifier': 'sqlite3',
                                            'relative_file_paths': ['sqlite_bag_0.db3'], 'compression_mode': '',
                                            'compression_format': ''}}
    (d / 'metadata.yaml').write_text(yaml.safe_dump(meta))
    with rb.BagReader(str(d)) as r:
        assert r.metadata['storage_identifier'] == 'sqlite3' and r.message_count == 600
    con = sqlite3.connect(str(tmp_bags / 'other.db3'))
    con.execute('CREATE TABLE x(a)')
    con.close()
    with pytest.raises(ValueError, match='not a rosbag2 SQLite bag'):
        rb.BagReader(str(tmp_bags / 'other.db3'))


def test_split_bag_merged(tmp_bags):
    """A bag of two .mcap files (as rosbag2 splits a long recording) reads as one time-ordered stream."""
    import yaml
    parts = []
    for k, offset in enumerate((0, 5)):
        p = str(tmp_bags / f'part{k}')
        with rb.BagWriter(p, compression='none') as bag:
            bag.add_topic('/n', 'std_msgs/msg/Int32')
            for i in range(10):
                bag.write('/n', offset + 10 * i, {'data': offset + 10 * i})
        parts.append(os.path.join(p, f'part{k}_0.mcap'))
    d = tmp_bags / 'split'
    d.mkdir()
    for k, p in enumerate(parts):
        shutil.move(p, d / f'split_{k}.mcap')
    meta = {'rosbag2_bagfile_information': {'version': 5, 'storage_identifier': 'mcap',
                                            'relative_file_paths': ['split_0.mcap', 'split_1.mcap']}}
    (d / 'metadata.yaml').write_text(yaml.safe_dump(meta))
    with rb.BagReader(str(d)) as r:
        assert r.counts == {'/n': 20} and len(r.files) == 2
        expect = sorted([10 * i for i in range(10)] + [5 + 10 * i for i in range(10)])
        assert [m['data'] for _, _, m in r.messages()] == expect


def test_truncated_and_corrupt_files(tmp_bags):
    path = str(tmp_bags / 'cut')
    with rb.BagWriter(path, compression='none', chunk_size=2048) as bag:
        bag.add_topic('/s', 'std_msgs/msg/String')
        for i in range(500):
            bag.write('/s', 1000 + i, {'data': f'message {i:04d}'})
        res = bag.close()
    full = open(res['file'], 'rb').read()
    for frac in (0.02, 0.3, 0.77, 0.999):
        cut = str(tmp_bags / f'cut_{frac}.mcap')
        with open(cut, 'wb') as f:
            f.write(full[:int(len(full) * frac)])
        t0 = time.time()
        with rb.BagReader(cut) as r:
            got = [m['data'] for _, _, m in r.messages()]
            assert r.warnings                                          # read by a linear scan, said so
        assert time.time() - t0 < 10
        assert got == [f'message {i:04d}' for i in range(len(got))]   # a clean prefix
        if frac > 0.5:
            assert len(got) > 300
    # a corrupt byte inside a chunk's messages is caught by the chunk's CRC
    bad = bytearray(full)
    bad[full.index(b'message 0100') + 3] ^= 0xff
    with open(str(tmp_bags / 'bad.mcap'), 'wb') as f:
        f.write(bad)
    with pytest.raises(ValueError):
        with rb.BagReader(str(tmp_bags / 'bad.mcap')) as r:
            list(r.messages())
    with open(str(tmp_bags / 'junk.mcap'), 'wb') as f:
        f.write(b'not a bag at all' * 10)
    with pytest.raises(ValueError):
        rb.BagReader(str(tmp_bags / 'junk.mcap'))
    with pytest.raises(FileNotFoundError):
        rb.BagReader(str(tmp_bags / 'missing'))


def test_writer_rules(tmp_bags):
    path = str(tmp_bags / 'rules')
    bag = rb.BagWriter(path, compression='none')
    bag.add_topic('/x', 'std_msgs/msg/Float64')
    bag.add_topic('/x', 'std_msgs/Float64')                            # the same again: fine
    with pytest.raises(ValueError):
        bag.add_topic('/x', 'std_msgs/msg/String')
    with pytest.raises(ValueError):
        bag.add_topic('/y', 'acme_msgs/msg/Nothing')
    with pytest.raises(ValueError):
        bag.write('/z', 0, {'data': 1.0})
    with pytest.raises(ValueError):
        bag.write('/x', -5, {'data': 1.0})
    with pytest.raises(ValueError):
        bag.write('/x', 5, b'\x07\x07')
    bag.write('/x', 5, {'data': 2.5})
    first = bag.close()
    assert bag.close() == first and first['messages'] == 1
    with pytest.raises(ValueError):
        bag.write('/x', 6, {'data': 1.0})
    with pytest.raises(FileExistsError):
        rb.BagWriter(path)
    with rb.BagWriter(path, overwrite=True) as again:
        again.add_topic('/tf_static', 'tf2_msgs/msg/TFMessage')
        again.write('/tf_static', 0, {'transforms': []})
    with rb.BagReader(path) as r:
        assert r.counts == {'/tf_static': 1}
    import yaml
    meta = yaml.safe_load(open(os.path.join(path, 'metadata.yaml')))['rosbag2_bagfile_information']
    qos = meta['topics_with_message_count'][0]['topic_metadata']['offered_qos_profiles']
    assert yaml.safe_load(qos)[0]['durability'] == 1                    # transient local, so playback latches it
    empty = str(tmp_bags / 'empty')
    with rb.BagWriter(empty) as bag:
        pass
    with rb.BagReader(empty) as r:
        assert r.message_count == 0 and r.start_ns == 0 and list(r.messages()) == []
    with pytest.raises(ValueError):
        rb.BagWriter(str(tmp_bags / 'c'), compression='brotli')


def test_write_speed(tmp_bags):
    """About 37 MB of camera frames are written in seconds, not minutes, and read back streaming (Vantage's 100 MB
    test, made smaller because many test runs share the machine's memory here)."""
    img = np.random.default_rng(0).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    path = str(tmp_bags / 'speed')
    t0 = time.perf_counter()
    with rb.BagWriter(path) as bag:
        bag.add_topic('/cam', 'sensor_msgs/msg/Image')
        for i in range(6):
            bag.write('/cam', i * 33_000_000, rb.image_msg(img, i * 33_000_000, 'cam'))
        res = bag.close()
    assert res['bytes'] > 37e6 and time.perf_counter() - t0 < 15
    t0 = time.perf_counter()
    with rb.BagReader(path) as r:
        n = sum(1 for _ in r.messages(decode=False))
    assert n == 6 and time.perf_counter() - t0 < 15


def test_skip_corrupt_chunk_and_fuzz(tmp_bags):
    path = str(tmp_bags / 'fz')
    with rb.BagWriter(path, compression='none', chunk_size=2048) as bag:
        bag.add_topic('/s', 'std_msgs/msg/String')
        for i in range(300):
            bag.write('/s', i, {'data': f'message {i:04d}'})
        res = bag.close()
    full = open(res['file'], 'rb').read()
    bad = bytearray(full)
    bad[full.index(b'message 0150') + 2] ^= 0x20
    fp = str(tmp_bags / 'one_bad.mcap')
    open(fp, 'wb').write(bad)
    with rb.BagReader(fp) as r:
        got = [m['data'] for _, _, m in r.messages(skip_corrupt=True)]
        assert 'message 0150' not in got and 250 < len(got) < 300            # one chunk left out
        assert got == sorted(got) and any('skipped' in w for w in r.warnings)
    # random damage: a reader either reads or raises ValueError, never anything else, never hangs
    rng = random.Random(5)
    for trial in range(60):
        b = bytearray(full)
        for _ in range(rng.randint(1, 8)):
            b[rng.randrange(len(b))] = rng.randrange(256)
        b = b[:rng.randrange(len(b) // 2, len(b) + 1)]
        open(fp, 'wb').write(b)
        try:
            with rb.BagReader(fp) as r:
                list(r.messages(skip_corrupt=True))
        except ValueError:
            pass


def test_rosbag2_message_compression_and_file_compression(tmp_bags):
    zstd = pytest.importorskip('zstandard', reason='zstandard (the bags extra) is not installed')
    import yaml
    d = tmp_bags / 'msgz'
    d.mkdir()
    rows = [(1, 100 + i, zstd.ZstdCompressor().compress(rb.encode('std_msgs/msg/Float64', {'data': i * 0.5})))
            for i in range(50)]
    _make_db3(str(d / 'msgz_0.db3'), rows, {1: ('/v', 'std_msgs/msg/Float64')})
    meta = {'rosbag2_bagfile_information': {'version': 5, 'storage_identifier': 'sqlite3',
                                            'relative_file_paths': ['msgz_0.db3'], 'compression_mode': 'MESSAGE',
                                            'compression_format': 'zstd'}}
    (d / 'metadata.yaml').write_text(yaml.safe_dump(meta))
    with rb.BagReader(str(d)) as r:
        assert [m['data'] for _, _, m in r.messages()] == [i * 0.5 for i in range(50)]
    meta['rosbag2_bagfile_information']['compression_mode'] = 'FILE'
    (d / 'metadata.yaml').write_text(yaml.safe_dump(meta))
    with pytest.raises(ValueError, match='decompress it first'):
        rb.BagReader(str(d))


def test_register_rules():
    with pytest.raises(ValueError, match='different layout'):
        rb.register('geometry_msgs/msg/Point', 'float32 x\nfloat32 y\nfloat32 z\n')
    rb.register('geometry_msgs/msg/Point', '# same layout, other comments\nfloat64 x\nfloat64 y\nfloat64 z\n')
    with pytest.raises(ValueError, match='unknown message type'):
        rb.register('dexvantage_test_msgs/msg/Orphan', 'dexvantage_test_msgs/Missing m\n')
    with pytest.raises(ValueError):
        rb.register('dexvantage_test_msgs/msg/Bad', 'float64[x] nope\n')
    rb.register('dexvantage_test_msgs/msg/Wide', 'wstring w\n')
    with pytest.raises(ValueError, match='wstring'):
        rb.encode('dexvantage_test_msgs/msg/Wide', {})


# ---------------------------------------------------------------------------------------------------------------------
# bags as datasets (core/datasets/bag.py)

from dexvantage.core import datasets as DS  # noqa: E402
from dexvantage.core.datasets import bag as BD  # noqa: E402
from dexvantage.core.frames import quat_to_matrix  # noqa: E402

T0 = 1_762_500_000_123_456_789          # an epoch time in 2025, in ns
JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']
K_WRIST = [[420.0, 0, 16.0], [0, 421.0, 12.0], [0, 0, 1]]
D_WRIST = [0.1, -0.02, 0.001, 0.0005, 0.003]
STATIC = [('base_link', 'scene_mount', [0.6, -0.3, 0.9], [0.0, 0.3826834, 0.0, 0.9238795]),
          ('scene_mount', 'scene_optical', [0.02, 0.0, 0.01], [-0.5, 0.5, -0.5, 0.5]),
          ('tool0', 'wrist_optical', [0.0, 0.05, 0.03], [0.0, 0.0, 0.0, 1.0])]


def _robot_messages(rng, n_joint=60, n_wrist=18, n_scene=9, device_offset_ns=-1_762_000_000_000_000_000):
    """(topic, type, log time, message dict) of a 0.6 s robot episode: joint_states at 100 Hz (header stamps 0.2 ms
    before the log time), a wrist rgb8 camera at 30 Hz with CameraInfo and a TimeReference giving a device clock that
    runs 50 ppm fast, a scene camera as JPEG at 15 Hz whose header stamps are on a device clock (seconds since power
    on), the static TF tree, a task string and a robot description."""
    import cv2
    out = []
    q = rng.normal(size=(n_joint, 6))
    for i in range(n_joint):
        st = T0 + i * 10_000_000
        out.append(('/joint_states', 'sensor_msgs/msg/JointState', st + 200_000,
                    {'header': rb.header(st, 'base_link'), 'name': JOINTS, 'position': q[i].tolist(),
                     'velocity': [0.0] * 6}))
    wrist = rng.integers(0, 256, (n_wrist, 24, 32, 3), dtype=np.uint8)
    wst = []
    for i in range(n_wrist):
        st = T0 + 3_000_000 + i * 33_333_333
        wst.append(st)
        out.append(('/wrist/color/image_raw', 'sensor_msgs/msg/Image', st + 4_000_000,
                    rb.image_msg(wrist[i], st, 'wrist_optical')))
        out.append(('/wrist/color/camera_info', 'sensor_msgs/msg/CameraInfo', st + 4_000_000,
                    rb.camera_info(32, 24, 'plumb_bob', D_WRIST, K_WRIST, t_ns=st, frame_id='wrist_optical')))
        dev = 5_000_000_000 + int(round((st - T0) * (1 + 50e-6)))
        out.append(('/wrist/color/time_reference', 'sensor_msgs/msg/TimeReference', st + 4_100_000,
                    {'header': rb.header(st, 'wrist_optical'), 'time_ref': rb.stamp(dev), 'source': 'uvc'}))
    smooth = np.zeros((n_scene, 24, 32, 3), np.uint8)
    sst = []
    for i in range(n_scene):
        smooth[i] = np.linspace(30, 220, 32, dtype=np.uint8)[None, :, None]
        smooth[i, :, :, 0] = 200
        log = T0 + 8_000_000 + i * 66_666_667
        sst.append(log)
        jpg = cv2.imencode('.jpg', cv2.cvtColor(smooth[i], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])[1]
        out.append(('/scene/image_raw/compressed', 'sensor_msgs/msg/CompressedImage', log,
                    rb.compressed_image(jpg.tobytes(), log + device_offset_ns, 'scene_optical')))
    out.append(('/scene/camera_info', 'sensor_msgs/msg/CameraInfo', T0 + 8_000_000,
                rb.camera_info(32, 24, 'equidistant', [0.01, 0.0, 0.0, 0.0], [[300.0, 0, 16], [0, 300, 12], [0, 0, 1]],
                               t_ns=T0, frame_id='scene_optical')))
    tfs = []
    for parent, child, p, qx in STATIC:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = quat_to_matrix(qx), p
        tfs.append(rb.transform_stamped(T, T0, parent, child))
    out.append(('/tf_static', 'tf2_msgs/msg/TFMessage', T0, {'transforms': tfs}))
    out.append(('/task', 'std_msgs/msg/String', T0, {'data': 'pick the red cube'}))
    out.append(('/robot_description', 'std_msgs/msg/String', T0, {'data': '<?xml version="1.0"?>\n<robot name="ur5e">'
                                                                          '<link name="base_link"/></robot>'}))
    sent = {'q': q, 'wrist': wrist, 'wrist_stamps': wst, 'scene': smooth, 'scene_log': sst,
            'scene_dev': [s + device_offset_ns for s in sst]}
    return out, sent


def _write_mcap_bag(path, msgs, compression='none'):
    with rb.BagWriter(path, compression=compression, chunk_size=1 << 16) as bag:
        for topic, typ, _, _ in msgs:
            bag.add_topic(topic, typ)
        for topic, _, t, m in msgs:
            bag.write(topic, t, m)


def _write_db3_bag(folder, msgs):
    """A rosbag2 SQLite bag written with sqlite3 and the schema rosbag2 creates (independent of the MCAP writer)."""
    os.makedirs(folder, exist_ok=True)
    types = {}
    for topic, typ, _, _ in msgs:
        types.setdefault(topic, typ)
    tid = {tp: i + 1 for i, tp in enumerate(types)}
    rows = [(tid[tp], t, rb.encode(typ, m)) for tp, typ, t, m in msgs]
    _make_db3(os.path.join(folder, os.path.basename(folder) + '_0.db3'), rows,
              {i: (tp, types[tp]) for tp, i in tid.items()})


def _expected_static(child):
    """The pose of `child` in its static root, composed by hand from the quaternions (not through rosbag)."""
    tree = {c: (p, np.array(pos), quat_to_matrix(qq)) for p, c, pos, qq in STATIC}
    R, t, cur = np.eye(3), np.zeros(3), child
    while cur in tree:
        p, pos, Rq = tree[cur]
        R, t = Rq @ R, Rq @ t + pos
        cur = p
    return cur, R, t


def _check_episode(ep, sent):
    n = len(sent['q'])
    assert ep.joint_names == JOINTS and ep.q.shape == (n, 6) and np.array_equal(ep.q, sent['q'])
    t0 = min(T0, sent['wrist_stamps'][0])
    assert ep.meta['t0_ns'] == t0 and ep.meta['t_source'] == 'header.stamp'
    assert np.array_equal(ep.t, (np.arange(n) * 10_000_000 + T0 - t0) / 1e9)
    assert np.allclose(ep.meta['log_times'], ep.t + 200e-6, atol=1e-12)
    assert ep.task == 'pick the red cube' and ep.robot_type == 'ur5e' and ep.action is None
    assert set(ep.cameras) == {'/wrist/color/image_raw', '/scene/image_raw/compressed'}
    w = ep.cameras['/wrist/color/image_raw']
    assert (w.width, w.height, w.n_frames) == (32, 24, len(sent['wrist']))
    assert np.array_equal(w.timestamps, (np.array(sent['wrist_stamps']) - t0) / 1e9)
    assert np.allclose(w.meta['log_times'] - w.timestamps, 4e-3, atol=1e-9)
    assert w.meta['timestamp_source'] == 'header.stamp'
    assert w.intrinsics['model'] == 'brown' and w.intrinsics['K'] == K_WRIST and w.intrinsics['dist'] == D_WRIST
    # the TimeReference device clock: 50 ppm fast, seconds from its first value
    expect_hw = (np.array(sent['wrist_stamps']) - sent['wrist_stamps'][0]) * (1 + 50e-6) / 1e9
    assert w.hw_timestamps is not None and np.allclose(w.hw_timestamps, expect_hw, atol=2e-9)
    assert 'TimeReference' in w.meta['hw_timestamp_source']
    root, R, tt = _expected_static('wrist_optical')
    assert w.extrinsics['frame'] == root == 'tool0' and np.allclose(w.extrinsics['pose']['position'], tt)
    assert np.allclose(quat_to_matrix(w.extrinsics['pose']['quaternion']), R, atol=1e-9)
    assert w.meta['tf_chain'] == ['tool0', 'wrist_optical']
    frames = dict(w.frames())
    assert len(frames) == len(sent['wrist']) and all(np.array_equal(frames[k], sent['wrist'][k]) for k in frames)
    assert set(dict(w.frames([5, 2]))) == {2, 5}
    with pytest.raises(DS.DatasetError, match='no frame'):
        list(w.frames([len(sent['wrist'])]))
    s = ep.cameras['/scene/image_raw/compressed']
    assert (s.width, s.height) == (32, 24) and s.intrinsics['model'] == 'kb'
    # its header stamps run on a device clock: they become hw_timestamps, timestamps are the log times
    assert 'log time' in s.meta['timestamp_source'] and 'device clock' in s.meta['hw_timestamp_source']
    log = np.array(sent['scene_log'])
    assert np.allclose(s.timestamps, (log - 200_000 - t0) / 1e9, atol=1e-9)      # onto the joint clock (-0.2 ms)
    assert np.allclose(s.hw_timestamps, (log - log[0]) / 1e9, atol=1e-9)
    root, R, tt = _expected_static('scene_optical')
    assert s.extrinsics['frame'] == root == 'base_link' and np.allclose(s.extrinsics['pose']['position'], tt)
    assert np.allclose(quat_to_matrix(s.extrinsics['pose']['quaternion']), R, atol=1e-9)
    img = dict(s.frames([3]))[3]
    assert img.shape == (24, 32, 3) and np.abs(img.astype(int) - sent['scene'][3]).mean() < 3
    assert abs(int(img[..., 0].mean()) - 200) < 4                      # red stays red: RGB order


@pytest.mark.parametrize('comp', _compressions())
def test_mcap_bag_as_episode(tmp_bags, comp):
    pytest.importorskip('cv2', reason='opencv-python (a core dependency) does not import here')
    msgs, sent = _robot_messages(np.random.default_rng(7))
    path = str(tmp_bags / 'ep_001')
    _write_mcap_bag(path, msgs, comp)
    ds = DS.open_dataset(path)
    assert isinstance(ds, BD.BagDataset) and ds.format == 'mcap' and ds.version == 'v5'
    refs = ds.episodes()
    assert [r.id for r in refs] == ['ep_001'] and refs[0].length == len(sent['q'])
    ep = ds.load(0)
    assert ep.id == 'ep_001' and not ep.meta['warnings']
    _check_episode(ep, sent)
    single = DS.open_dataset(os.path.join(path, 'ep_001_0.mcap'))       # the .mcap file alone
    _check_episode(single.load(0), sent)
    assert ds.truth() is None


def test_sqlite_bag_as_episode(tmp_bags):
    pytest.importorskip('cv2', reason='opencv-python (a core dependency) does not import here')
    msgs, sent = _robot_messages(np.random.default_rng(8))
    folder = str(tmp_bags / 'ep_db3')
    _write_db3_bag(folder, msgs)
    ds = DS.open_dataset(folder)
    assert ds.format == 'rosbag2-sqlite'
    _check_episode(ds.load(0), sent)


def test_bag_written_by_the_mcap_package(tmp_bags):
    """An MCAP file from mcap.writer (schemas from rosbag's schema text, messages encoded by hand-checked CDR)."""
    pytest.importorskip('mcap', reason='mcap (the test extra, an independent MCAP reader) is not installed')
    pytest.importorskip('cv2', reason='opencv-python (a core dependency) does not import here')
    from mcap.writer import CompressionType, Writer
    msgs, sent = _robot_messages(np.random.default_rng(9))
    path = str(tmp_bags / 'theirs.mcap')
    with open(path, 'wb') as f:
        w = Writer(f, compression=CompressionType.NONE)
        w.start(profile='ros2', library='test')
        sids, cids = {}, {}
        for topic, typ, _, _ in msgs:
            if typ not in sids:
                sids[typ] = w.register_schema(name=typ, encoding='ros2msg', data=rb.schema_text(typ).encode())
            if topic not in cids:
                cids[topic] = w.register_channel(topic=topic, message_encoding='cdr', schema_id=sids[typ])
        for topic, typ, t, m in sorted(msgs, key=lambda x: x[2]):
            w.add_message(cids[topic], log_time=t, data=rb.encode(typ, m), publish_time=t)
        w.finish()
    _check_episode(DS.open_dataset(path).load(0), sent)


def test_folder_of_bags_is_one_episode_each(tmp_bags):
    rng = np.random.default_rng(10)
    root = tmp_bags / 'campaign'
    root.mkdir()
    for name in ('ep_10', 'ep_2'):
        _write_mcap_bag(str(root / name), _robot_messages(rng, 20, 2, 1)[0])
    loose = tmp_bags / 'loose'
    _write_mcap_bag(str(loose), _robot_messages(rng, 30, 2, 1)[0])
    shutil.move(str(loose / 'loose_0.mcap'), str(root / 'ep_3.mcap'))
    (root / 'notes.txt').write_text('not a bag')
    ds = DS.open_dataset(str(root))
    assert [r.id for r in ds.episodes()] == ['ep_2', 'ep_3', 'ep_10']      # natural order
    assert [len(ds.load(i).t) for i in range(3)] == [20, 30, 20]
    with pytest.raises(DS.DatasetError, match='no episode 4'):
        ds.load(3)
    doc = {'schema': 'dexvantage.truth/1', 'episodes': {}}
    (root / 'dexvantage_truth.json').write_text(json.dumps(doc))
    assert ds.truth() == doc


def test_joint_subsets_unstamped_and_depth(tmp_bags):
    """A driver that publishes the gripper on its own, leaves header stamps at zero, and a depth topic."""
    path = str(tmp_bags / 'odd')
    with rb.BagWriter(path, compression='none') as bag:
        bag.add_topic('/joint_states', 'sensor_msgs/msg/JointState')
        bag.add_topic('/depth/image_rect_raw', 'sensor_msgs/msg/Image')
        bag.add_topic('/arm/command', 'sensor_msgs/msg/JointState')
        for i in range(6):
            t = 1000 + i * 1_000_000
            if i % 2 == 0:
                msg = {'name': ['a', 'b'], 'position': [i, -i]}
            else:
                msg = {'name': ['grip'], 'position': [10.0 * i]}
            bag.write('/joint_states', t, msg)
            bag.write('/arm/command', t, {'name': ['a'], 'position': [0.0]})
        bag.write('/depth/image_rect_raw', 2000, rb.image_msg(np.zeros((4, 5), np.uint16), 2000, 'depth', '16UC1'))
    ep = DS.open_dataset(path).load(0)
    assert ep.joint_names == ['a', 'b', 'grip'] and ep.meta['t_source'] == 'log time'
    assert np.allclose(ep.t, np.arange(6) * 1e-3)
    assert np.isnan(ep.q[0, 2]) and ep.q[1].tolist() == [0, 0, 10.0] and ep.q[2].tolist() == [2, -2, 10.0]
    assert ep.q[3].tolist() == [2, -2, 30.0] and ep.meta['q_held'] == 2 + 1 + 2 + 1 + 2
    assert ep.meta['other_joint_topics'] == ['/arm/command']
    assert ep.cameras == {} and ep.meta['depth_streams']['/depth/image_rect_raw']['encoding'] == '16UC1'


def test_pixel_encodings_to_rgb():
    cv2 = pytest.importorskip('cv2', reason='opencv-python (a core dependency) does not import here')
    rng = np.random.default_rng(4)
    rgb = rng.integers(0, 256, (6, 8, 3), dtype=np.uint8)
    assert np.array_equal(BD.to_rgb8(rgb[:, :, ::-1], 'bgr8'), rgb)
    assert np.array_equal(BD.to_rgb8(rgb[:, :, 1], 'mono8'), np.repeat(rgb[:, :, 1:2], 3, 2))
    rgba = np.concatenate([rgb, np.full((6, 8, 1), 9, np.uint8)], 2)
    assert np.array_equal(BD.to_rgb8(rgba[:, :, [2, 1, 0, 3]], 'bgra8'), rgb)
    assert np.array_equal(BD.to_rgb8(rgb.astype(np.uint16) << 8, 'rgb16'), rgb)
    # an RGGB mosaic of a flat colour demosaics back to that colour (cv_bridge's mapping of ROS names to OpenCV's)
    flat = np.zeros((8, 8), np.uint8)
    flat[0::2, 0::2], flat[0::2, 1::2], flat[1::2, 0::2], flat[1::2, 1::2] = 200, 100, 100, 30
    out = BD.to_rgb8(flat, 'bayer_rggb8')
    assert np.abs(out[2:-2, 2:-2].reshape(-1, 3).mean(0) - [200, 100, 30]).max() < 1
    assert cv2 is not None


def test_damaged_and_unreadable_bags_are_explained(tmp_bags, monkeypatch):
    with pytest.raises(DS.DatasetError, match='no dataset'):
        DS.open_dataset(str(tmp_bags / 'missing'))
    junk = tmp_bags / 'junk.mcap'
    junk.write_bytes(b'nope' * 10)
    assert not BD.episode_paths(str(junk))
    msgs, _ = _robot_messages(np.random.default_rng(11), 30, 3, 2)
    path = str(tmp_bags / 'bad')
    _write_mcap_bag(path, msgs)
    fp = os.path.join(path, 'bad_0.mcap')
    blob = bytearray(open(fp, 'rb').read())
    blob[len(blob) // 3] ^= 0xFF
    open(fp, 'wb').write(blob)
    with pytest.raises(DS.DatasetError) as e:
        DS.open_dataset(path).load(0)
    text = str(e.value)
    assert 'damaged' in text and str(tmp_bags) not in text and 'Traceback' not in text
    zpath = str(tmp_bags / 'z')
    if 'zstd' in _compressions():
        _write_mcap_bag(zpath, msgs, 'zstd')
        monkeypatch.setitem(sys.modules, 'zstandard', None)          # as if the bags extra were not installed
        with pytest.raises(DS.DatasetError, match='bags extra'):
            DS.open_dataset(zpath).load(0)


def test_mcap_roundtrip_check_passes():
    from dexvantage.validation import references as VR
    from dexvantage.validation import suite
    suite.load_checks()
    spec = next(s for s in suite.CHECKS if s.id == 'formats.mcap_roundtrip')
    assert spec.refs and all(k in VR.REFERENCES for k in spec.refs)
    ctx = suite.Ctx(None, None, None, suite.ValidationSettings(models=('formats',), quick=True))
    try:
        c = suite.run_one(spec, ctx)
    finally:
        ctx.close()
    assert c.passed, (c.error, c.detail)
    json.dumps(c.detail)
    assert c.detail['messages'] > 60 and c.detail['spec_walk_crc_failures'] == 0 and c.detail['cdr_hand_layout_equal']


def test_mcap_roundtrip_check_catches_a_slip(monkeypatch):
    """A reader that dropped the nanoseconds of a stamp must fail the check."""
    from dexvantage.validation import suite
    suite.load_checks()
    monkeypatch.setattr(rb, 'stamp_ns', lambda st: int(st.get('sec', 0)) * 1_000_000_000)
    spec = next(s for s in suite.CHECKS if s.id == 'formats.mcap_roundtrip')
    ctx = suite.Ctx(None, None, None, suite.ValidationSettings(models=('formats',), quick=True))
    try:
        c = suite.run_one(spec, ctx)
    finally:
        ctx.close()
    assert not c.passed and c.detail['reader_mismatches'] > 0
