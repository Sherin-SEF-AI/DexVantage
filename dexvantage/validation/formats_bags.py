"""The work behind the checks formats.mcap_roundtrip and formats.tfrecord_crc (registered in checks_formats.py).

formats.mcap_roundtrip: a robot episode (joint_states, an rgb8 and a PNG camera with CameraInfo, the static TF tree)
is written as a ROS 2 bag by core/rosbag.py's BagWriter with every chunk compression installed, then read back three
ways that share no code with the writer: (1) a walk of the file written here again from the MCAP specification
(https://mcap.dev/spec, read 2026-11-07): magic at both ends, every record's opcode and length, each chunk decompressed
with zstandard / lz4 directly and its uncompressed CRC-32 checked, the data-section CRC in DataEnd and the summary CRC
in the Footer; (2) the mcap package (Foxglove's reference reader) with CRC validation, when installed; (3) one
JointState's CDR bytes laid out by hand with struct (XCDR1: 4-byte encapsulation header, each primitive aligned to its
size from the end of the header, a string as uint32 length counting the NUL); and DexVantage's dataset reader
(core/datasets/bag.py) must give back the joint states and stamps exactly. The error is the number of mismatches.

formats.tfrecord_crc: core/datasets/rlds.py's CRC-32C against the five published vectors of RFC 3720 appendix B.4
and the CRC catalogue's check value (crc32c('123456789') = 0xe3069283), against a bit-at-a-time Castagnoli CRC
written here on random buffers on both sides of the vectorised path's threshold, its masked form against TensorFlow's
formula (xla/tsl/lib/hash/crc32c.h), and TFRecord framing written here and read by the reader, with a damaged copy
that must be refused. The error is the number of mismatches.
"""
from __future__ import annotations

import os
import struct
import zlib

import numpy as np

# ---------------------------------------------------------------------------------------------------------------------
# MCAP

JOINTS = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
T0 = 1_762_500_000_000_000_000


def reference_messages(rng, quick: bool):
    """(topic, type, log time ns, message) of a short robot episode."""
    import cv2

    from ..core import rosbag as RB
    n_j, n_c = (40, 8) if quick else (200, 40)
    out = []
    q = rng.normal(size=(n_j, 6))
    for i in range(n_j):
        st = T0 + i * 10_000_000
        out.append(('/joint_states', 'sensor_msgs/msg/JointState', st + 150_000,
                    {'header': RB.header(st, 'base_link'), 'name': JOINTS, 'position': q[i].tolist()}))
    for i in range(n_c):
        st = T0 + 2_000_000 + i * 33_333_333
        img = rng.integers(0, 256, (12, 16, 3), dtype=np.uint8)
        out.append(('/wrist/image_raw', 'sensor_msgs/msg/Image', st + 3_000_000,
                    RB.image_msg(img, st, 'wrist_optical')))
        out.append(('/wrist/camera_info', 'sensor_msgs/msg/CameraInfo', st + 3_000_000,
                    RB.camera_info(16, 12, 'plumb_bob', [0.01, 0, 0, 0, 0], [[20.0, 0, 8], [0, 20, 6], [0, 0, 1]],
                                   t_ns=st, frame_id='wrist_optical')))
        png = cv2.imencode('.png', img)[1].tobytes()
        out.append(('/scene/image_raw/compressed', 'sensor_msgs/msg/CompressedImage', st + 5_000_000,
                    RB.compressed_image(png, st, 'scene_optical', 'png')))
    T = np.eye(4)
    T[:3, 3] = [0.5, 0.0, 1.0]
    out.append(('/tf_static', 'tf2_msgs/msg/TFMessage', T0, {'transforms': [RB.transform_stamped(T, T0, 'base_link',
                                                                                                 'scene_optical')]}))
    return out, q


def joint_state_cdr_by_hand(sec: int, nsec: int, frame: str, names: list, pos: list) -> bytes:
    """sensor_msgs/JointState in XCDR1 little-endian, laid out field by field (velocity and effort empty)."""
    b = bytearray(struct.pack('<iI', sec, nsec))
    fb = frame.encode() + b'\0'
    b += struct.pack('<I', len(fb)) + fb
    b += bytes(-len(b) % 4) + struct.pack('<I', len(names))
    for n in names:
        nb = n.encode() + b'\0'
        b += bytes(-len(b) % 4) + struct.pack('<I', len(nb)) + nb
    b += bytes(-len(b) % 4) + struct.pack('<I', len(pos))
    b += bytes(-len(b) % 8) + struct.pack(f'<{len(pos)}d', *pos)
    b += bytes(-len(b) % 4) + struct.pack('<I', 0)          # velocity: empty
    b += struct.pack('<I', 0)                               # effort: empty
    return b'\x00\x01\x00\x00' + bytes(b)


def spec_walk(path: str) -> tuple[list, int, list]:
    """([(topic, log time, data)], CRC failures, problems) of an MCAP file, from the specification alone."""
    data = open(path, 'rb').read()
    magic = b'\x89MCAP0\r\n'
    problems, crc_bad = [], 0
    if data[:8] != magic or data[-8:] != magic:
        return [], 1, ['magic missing']
    channels, schemas, msgs = {}, {}, []

    def string(buf, p):
        n, = struct.unpack_from('<I', buf, p)
        return buf[p + 4:p + 4 + n].decode(), p + 4 + n

    def records(buf, p, end):
        while p < end:
            op, ln = struct.unpack_from('<BQ', buf, p)
            yield op, p, p + 9, p + 9 + ln
            p += 9 + ln

    def inner(buf):
        for op, _, s, e in records(buf, 0, len(buf)):
            if op == 0x03:                                   # Schema
                sid, = struct.unpack_from('<H', buf, s)
                schemas[sid] = string(buf, s + 2)[0]
            elif op == 0x04:                                 # Channel
                cid, sid = struct.unpack_from('<HH', buf, s)
                channels[cid] = string(buf, s + 4)[0]
            elif op == 0x05:                                 # Message
                cid, seq, log, pub = struct.unpack_from('<HIQQ', buf, s)
                msgs.append((cid, log, bytes(buf[s + 22:e])))

    p, end = 8, len(data) - 8
    for op, start, s, e in records(data, p, end):
        if op in (0x03, 0x04, 0x05):
            inner(data[start:e])
        elif op == 0x06:                                     # Chunk
            t0, t1, usize, ucrc = struct.unpack_from('<QQQI', data, s)
            comp, q = string(data, s + 28)
            rl, = struct.unpack_from('<Q', data, q)
            raw = data[q + 8:q + 8 + rl]
            if comp == 'zstd':
                import zstandard
                raw = zstandard.ZstdDecompressor().decompress(raw, max_output_size=usize)
            elif comp == 'lz4':
                import lz4.frame
                raw = lz4.frame.decompress(raw)
            elif comp:
                problems.append(f'unknown compression {comp}')
                continue
            if len(raw) != usize or (ucrc and zlib.crc32(raw) != ucrc):
                crc_bad += 1
            inner(raw)
        elif op == 0x0F:                                     # DataEnd: CRC of everything before it
            dcrc, = struct.unpack_from('<I', data, s)
            if dcrc and zlib.crc32(data[:start]) != dcrc:
                crc_bad += 1
        elif op == 0x02:                                     # Footer
            s_start, so_start, scrc = struct.unpack_from('<QQI', data, s)
            if scrc and s_start and zlib.crc32(data[s_start:s + 16]) != scrc:
                crc_bad += 1
            break
    return [(channels.get(c, f'#{c}'), t, d) for c, t, d in msgs], crc_bad, problems


def mcap_roundtrip(ctx) -> tuple[int, dict]:
    from ..core import rosbag as RB
    from ..core.datasets import open_dataset
    rng = ctx.rng('formats.mcap_roundtrip')
    msgs, q = reference_messages(rng, ctx.quick)
    sent = sorted(((tp, t, RB.encode(ty, m)) for tp, ty, t, m in msgs), key=lambda x: (x[1], x[0]))
    comps = ['none']
    for name, mod in (('zstd', 'zstandard'), ('lz4', 'lz4.frame')):
        try:
            __import__(mod)
            comps.append(name)
        except ImportError:
            pass
    try:
        import mcap
        from mcap.reader import make_reader
        mcap_version = getattr(mcap, '__version__', 'installed')
    except ImportError:
        make_reader, mcap_version = None, None
    detail = {'messages': len(sent), 'compressions': comps, 'spec_walk_crc_failures': 0, 'spec_walk_mismatches': 0,
              'mcap_package': mcap_version, 'mcap_package_mismatches': None, 'reader_mismatches': 0,
              'problems': []}
    # one JointState's CDR bytes against the hand layout
    m0 = msgs[0][3]
    hand = joint_state_cdr_by_hand(m0['header']['stamp']['sec'], m0['header']['stamp']['nanosec'], 'base_link',
                                   JOINTS, m0['position'])
    cdr_ok = RB.encode('sensor_msgs/msg/JointState', m0) == hand
    detail['cdr_hand_layout_equal'] = cdr_ok
    err = 0 if cdr_ok else 1
    for comp in comps:
        path = os.path.join(ctx.tmp(), f'mcap_rt_{comp}')
        with RB.BagWriter(path, compression=comp, chunk_size=1 << 14, overwrite=True) as bag:
            for tp, ty, _, _ in msgs:
                bag.add_topic(tp, ty)
            for tp, ty, t, m in msgs:
                bag.write(tp, t, m)
        fp = os.path.join(path, f'mcap_rt_{comp}_0.mcap')
        got, crc_bad, probs = spec_walk(fp)
        got = sorted(got, key=lambda x: (x[1], x[0]))
        mism = sum(1 for a, b in zip(got, sent) if a != b) + abs(len(got) - len(sent))
        detail['spec_walk_crc_failures'] += crc_bad
        detail['spec_walk_mismatches'] += mism
        detail['problems'] += [f'{comp}: {p}' for p in probs]
        err += crc_bad + mism + len(probs)
        if make_reader is not None:
            with open(fp, 'rb') as f:
                rd = make_reader(f, validate_crcs=True)
                theirs = sorted(((ch.topic, m.log_time, bytes(m.data)) for _, ch, m in rd.iter_messages()),
                                key=lambda x: (x[1], x[0]))
            mm = sum(1 for a, b in zip(theirs, sent) if a != b) + abs(len(theirs) - len(sent))
            detail['mcap_package_mismatches'] = (detail['mcap_package_mismatches'] or 0) + mm
            err += mm
        ep = open_dataset(path).load(0)
        stamps = np.array([T0 + i * 10_000_000 for i in range(len(q))])
        bad = int(ep.joint_names != JOINTS) + int(ep.q.shape != q.shape or not np.array_equal(ep.q, q)) + \
            int(not np.array_equal(ep.t, (stamps - stamps[0]) / 1e9))
        cam = ep.cameras.get('/wrist/image_raw')
        bad += int(cam is None or cam.extrinsics is not None or cam.intrinsics is None)
        sc = ep.cameras.get('/scene/image_raw/compressed')
        bad += int(sc is None or sc.extrinsics is None or
                   not np.allclose(sc.extrinsics['pose']['position'], [0.5, 0.0, 1.0]))
        detail['reader_mismatches'] += bad
        err += bad
    return err, detail


# ---------------------------------------------------------------------------------------------------------------------
# CRC-32C

RFC3720_B4 = [('32 bytes of zeroes', bytes(32), 0x8A9136AA), ('32 bytes of ones', b'\xff' * 32, 0x62A8AB43),
              ('32 bytes incrementing 00..1f', bytes(range(32)), 0x46DD794E),
              ('32 bytes decrementing 1f..00', bytes(range(31, -1, -1)), 0x113FDB5C),
              ('an iSCSI SCSI Read (10) command PDU',
               bytes.fromhex('01c0000000000000000000000000000014000000000004000000001400000018'
                             '280000000000000002000000' '00000000'), 0xD9963A56)]


def crc32c_bitwise(data: bytes) -> int:
    c = 0xFFFFFFFF
    for b in data:
        c ^= b
        for _ in range(8):
            c = (c >> 1) ^ (0x82F63B78 if c & 1 else 0)
    return c ^ 0xFFFFFFFF


def tfrecord_crc(ctx) -> tuple[int, dict]:
    from ..core.datasets import rlds as R
    rng = ctx.rng('formats.tfrecord_crc')
    rows, err = [], 0
    for name, data, crc in RFC3720_B4:
        got = R.crc32c(data)
        rows.append({'vector': name, 'expected': f'{crc:08x}', 'got': f'{got:08x}'})
        err += int(got != crc)
    check_value = R.crc32c(b'123456789')
    err += int(check_value != 0xE3069283)
    sizes = [1, 3, 100, 8191, 8192, 12_345] if ctx.quick else [1, 3, 100, 4095, 8191, 8192, 12_345, 65_537, 150_001]
    random_bad = 0
    for n in sizes:
        d = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
        random_bad += int(R.crc32c(d) != crc32c_bitwise(d))
    err += random_bad
    mask_bad = 0
    for c in rng.integers(0, 2 ** 32, 200, dtype=np.uint64).tolist():
        c = int(c)
        ref = ((((c >> 15) | (c << 17)) & 0xFFFFFFFF) + 0xA282EAD8) & 0xFFFFFFFF
        mask_bad += int(R.mask_crc(c) != ref or R.unmask_crc(ref) != c)
    err += mask_bad
    # framing written here, read by the reader; a damaged copy refused
    payloads = [rng.integers(0, 256, int(k), dtype=np.uint8).tobytes() for k in (0, 17, 5000, 20_000)]

    def frame(p):
        h = struct.pack('<Q', len(p))
        m = lambda c: ((((c >> 15) | (c << 17)) & 0xFFFFFFFF) + 0xA282EAD8) & 0xFFFFFFFF   # noqa: E731
        return h + struct.pack('<I', m(crc32c_bitwise(h))) + p + struct.pack('<I', m(crc32c_bitwise(p)))
    path = os.path.join(ctx.tmp(), 'crc_check.tfrecord')
    blob = b''.join(frame(p) for p in payloads)
    with open(path, 'wb') as f:
        f.write(blob)
    try:
        framing_bad = int(R.read_tfrecords(path) != payloads)
    except R.DatasetError:
        framing_bad = 1                                      # good records refused
    bad = bytearray(blob)
    bad[len(frame(payloads[0])) + len(frame(payloads[1])) + 12 + 2500] ^= 0x10
    with open(path, 'wb') as f:
        f.write(bad)
    try:
        R.read_tfrecords(path)
        framing_bad += 1                                     # the damage went unnoticed
    except R.DatasetError:
        pass
    err += framing_bad
    return err, {'rfc3720_vectors': len(RFC3720_B4), 'vectors': rows, 'check_value': f'{check_value:08x}',
                 'random_sizes': sizes, 'random_mismatches': random_bad, 'mask_mismatches': mask_bad,
                 'framing_mismatches': framing_bad}
