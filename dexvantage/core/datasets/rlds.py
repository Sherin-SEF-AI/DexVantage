"""RLDS / Open X-Embodiment datasets without TensorFlow: TFRecord framing with its CRC-32C checks, tf.train.Example and
tf.train.SequenceExample decoded by hand from the protobuf wire format, TFDS's features.json / dataset_info.json for
feature names, shapes and encodings, and each episode's steps mapped onto the shared Episode shape (episode.py).

Format, as read from the sources on 2026-11-07:

- TFRecord (TensorFlow, xla/tsl/lib/io/record_writer.h, https://github.com/openxla/xla, and
  https://www.tensorflow.org/tutorials/load_data/tfrecord): records concatenated, each
  `uint64 length | uint32 masked_crc32c(length) | data[length] | uint32 masked_crc32c(data)`, little-endian, with
  masked(c) = ((c >> 15) | (c << 17)) + 0xa282ead8 (mod 2^32) (xla/tsl/lib/hash/crc32c.h). CRC-32C is the Castagnoli
  CRC (reflected polynomial 0x82f63b78, init and xorout 0xffffffff), whose published test vectors are RFC 3720 B.4.
  Both CRCs of every record read are verified; a GZIP-compressed TFRecord file is read through Python's gzip module.
- tf.train.Example / SequenceExample (tensorflow/core/example/example.proto and feature.proto,
  https://github.com/tensorflow/tensorflow): Example{Features features = 1}; SequenceExample{Features context = 1;
  FeatureLists feature_lists = 2}; Features{map<string, Feature> feature = 1}; Feature{oneof: BytesList bytes_list = 1,
  FloatList float_list = 2, Int64List int64_list = 3}; FloatList and Int64List are packed (an unpacked encoding is read
  too, as the protobuf encoding guide requires of a parser, https://protobuf.dev/programming-guides/encoding/).
- TFDS serialisation (tensorflow_datasets/core/example_serializer.py, https://github.com/tensorflow/datasets): the
  nested feature dict is flattened with '/' ('steps/observation/image'); a one-level sequence (the RLDS 'steps'
  Dataset) is stacked and stored flat, so a (3,) float32 action over N steps is a FloatList of 3N values, an image is
  one encoded JPEG/PNG per step in a BytesList, bool is stored as int64, a Tensor with encoding 'bytes' / 'zlib' is
  one (zlib-compressed) raw buffer per step. features.json gives each leaf's class, shape, dtype and encoding.
- RLDS (https://github.com/google-research/rlds, README "Dataset Format"; Ramos et al. 2021, arXiv:2111.02767): an
  episode holds 'steps' (is_first, is_last, is_terminal, observation, action, reward, discount, ...) and
  episode_metadata.

Mapping onto Episode: one record = one episode; q is the observation's state vector (the first of joint_position(s),
joint_pos, joint_state(s), qpos, state, robot_state, proprio that exists; its components are named '<feature>_<i>'
because RLDS stores no joint names), action is the 'action' tensor or the 'action/*' sub-features concatenated in
features.json order, task is the language instruction, every 8-bit image feature is a CameraStream decoded lazily
with OpenCV. Times: RLDS stores none in general, so t is the step index divided by the fps given to the reader (the
fps is a user setting, never a guess; without it t counts steps and meta['time_base'] says 'steps'), unless the steps
have a float feature named 'timestamp', which is read as seconds (said in meta['time_source']).

Known limits: float64 tensors stored by TFDS as a FloatList keep float32 precision; float32 and uint16 images (depth)
are listed in meta['depth_streams'] and not exposed as cameras; ragged (multi-level) sequences are left out with a
note; intrinsics and extrinsics are None (RLDS has no standard place for them); a gzip TFRecord is read from the start
for every episode, so it is slow to browse.
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import re
import struct
import zlib

import numpy as np

from .episode import CameraStream, Dataset, DatasetError, Episode, EpisodeRef, register_reader

# ---------------------------------------------------------------------------------------------------------------------
# CRC-32C (Castagnoli), vectorised with numpy

CRC32C_POLY = 0x82F63B78          # reflected form of 0x1EDC6F41
MASK_DELTA = 0xA282EAD8


def _make_table() -> np.ndarray:
    t = np.arange(256, dtype=np.uint32)
    for _ in range(8):
        t = np.where(t & 1, (t >> 1) ^ np.uint32(CRC32C_POLY), t >> 1).astype(np.uint32)
    return t


_TABLE = _make_table()
_TABLE_L = _TABLE.tolist()
_SHIFT_CACHE: dict = {}


def _crc_bytes(reg: int, data) -> int:
    """The reflected CRC register after `data`, byte by byte (short inputs)."""
    t = _TABLE_L
    for b in data:
        reg = t[(reg ^ b) & 0xFF] ^ (reg >> 8)
    return reg


def _shift_tables(n: int):
    """Four 256-entry tables of the linear map 'register after n zero bytes' (one per register byte)."""
    hit = _SHIFT_CACHE.get(n)
    if hit is None:
        regs = (np.uint32(1) << np.arange(32, dtype=np.uint32)).astype(np.uint32)
        for _ in range(n):
            regs = _TABLE[regs & 0xFF] ^ (regs >> 8)
        bits = ((np.arange(256)[:, None] >> np.arange(8)[None, :]) & 1).astype(bool)
        hit = []
        for k in range(4):
            img = regs[8 * k:8 * k + 8]
            hit.append([int(np.bitwise_xor.reduce(img[bits[v]], initial=np.uint32(0))) for v in range(256)])
        _SHIFT_CACHE[n] = hit
    return hit


def crc32c(data, crc: int = 0) -> int:
    """CRC-32C of `data` (bytes-like), continuing from a previous value `crc` (0 for a fresh one): the iSCSI /
    Castagnoli CRC, crc32c(b'123456789') == 0xe3069283. Long inputs are split into equal blocks whose registers advance
    together in numpy and are then combined with the 'n zero bytes' linear map, so 100 MB take about a second."""
    buf = np.frombuffer(memoryview(data).cast('B'), np.uint8) if not isinstance(data, np.ndarray) else \
        np.ascontiguousarray(data, np.uint8).reshape(-1)
    reg = (crc ^ 0xFFFFFFFF) & 0xFFFFFFFF
    n = buf.size
    if n < 8192:
        return _crc_bytes(reg, buf.tobytes()) ^ 0xFFFFFFFF
    L = 1 << max(9, int(np.ceil(np.log2(np.sqrt(n)))))     # block length, about sqrt(n)
    B = n // L
    blocks = np.ascontiguousarray(buf[:B * L].reshape(B, L).T)  # (L, B): byte j of every block
    regs = np.zeros(B, np.uint32)
    for j in range(L):
        regs = _TABLE[(regs ^ blocks[j]) & 0xFF] ^ (regs >> 8)
    t0, t1, t2, t3 = _shift_tables(L)
    for c in regs.tolist():
        reg = (t0[reg & 0xFF] ^ t1[(reg >> 8) & 0xFF] ^ t2[(reg >> 16) & 0xFF] ^ t3[reg >> 24]) ^ c
    reg = _crc_bytes(reg, buf[B * L:].tobytes())
    return reg ^ 0xFFFFFFFF


def mask_crc(crc: int) -> int:
    """TFRecord's masked CRC (xla/tsl/lib/hash/crc32c.h Mask): rotate right by 15 bits, add 0xa282ead8."""
    crc &= 0xFFFFFFFF
    return ((((crc >> 15) | (crc << 17)) & 0xFFFFFFFF) + MASK_DELTA) & 0xFFFFFFFF


def unmask_crc(masked: int) -> int:
    rot = (masked - MASK_DELTA) & 0xFFFFFFFF
    return ((rot >> 17) | (rot << 15)) & 0xFFFFFFFF


# ---------------------------------------------------------------------------------------------------------------------
# TFRecord framing

class TFRecordFile:
    """One TFRecord file: index() lists (offset, length) of its records from their headers (length CRC verified,
    data skipped); read(offset) returns one record's data with its data CRC verified. GZIP files are read through
    gzip (offsets in the decompressed stream)."""

    def __init__(self, path: str):
        self.path = path
        with open(path, 'rb') as f:
            self.gzip = f.read(2) == b'\x1f\x8b'
        self._index = None

    def _open(self):
        return gzip.open(self.path, 'rb') if self.gzip else open(self.path, 'rb')

    def _where(self, k=None, start=False):
        name = os.path.basename(self.path)
        return f'{"Record" if start else "record"} {k + 1} of {name}' if k is not None else name

    def index(self) -> list:
        if self._index is not None:
            return self._index
        out = []
        with self._open() as f:
            pos = 0
            while True:
                head = f.read(12)
                if not head:
                    break
                if len(head) < 12:
                    raise DatasetError(f'{self._where()} ends inside a record header (the file is truncated).')
                n, mcrc = struct.unpack('<QI', head)
                if mask_crc(crc32c(head[:8])) != mcrc:
                    raise DatasetError(f'The length of {self._where(len(out))} is damaged (its checksum does not '
                                       'match), so the file cannot be read past it.')
                if self.gzip:
                    left = n + 4
                    while left:
                        got = len(f.read(min(left, 1 << 24)))
                        if not got:
                            break
                        left -= got
                    if left:
                        raise DatasetError(f'{self._where()} ends inside {self._where(len(out))} (truncated).')
                else:
                    f.seek(n + 4, 1)
                    if f.tell() > os.path.getsize(self.path):
                        raise DatasetError(f'{self._where()} ends inside {self._where(len(out))} (truncated).')
                out.append((pos, n))
                pos += 12 + n + 4
        self._index = out
        return out

    def read(self, offset: int, k: int | None = None) -> bytes:
        with self._open() as f:
            f.seek(offset)
            head = f.read(12)
            if len(head) < 12:
                raise DatasetError(f'{self._where(k, True)} is missing (the file is shorter than its index).')
            n, mcrc = struct.unpack('<QI', head)
            if mask_crc(crc32c(head[:8])) != mcrc:
                raise DatasetError(f'The length of {self._where(k)} is damaged (its checksum does not match).')
            data = f.read(n)
            tail = f.read(4)
        if len(data) < n or len(tail) < 4:
            raise DatasetError(f'{self._where(k, True)} is cut short (the file is truncated).')
        if mask_crc(crc32c(data)) != struct.unpack('<I', tail)[0]:
            raise DatasetError(f'{self._where(k, True)} is damaged: its data checksum does not match, so it was not '
                               'read.')
        return data

    def records(self):
        for k, (off, _) in enumerate(self.index()):
            yield self.read(off, k)


def read_tfrecords(path: str) -> list:
    """Every record of a TFRecord file (CRCs verified): a convenience for small files and tests."""
    return list(TFRecordFile(path).records())


# ---------------------------------------------------------------------------------------------------------------------
# protobuf wire format: tf.train.Example / SequenceExample

def _varint(b, p: int):
    v = shift = 0
    while True:
        if p >= len(b):
            raise ValueError('a varint runs past the end of the message')
        c = b[p]
        p += 1
        v |= (c & 0x7F) << shift
        if c < 0x80:
            return v, p
        shift += 7
        if shift > 70:
            raise ValueError('a varint is longer than 10 bytes')


def _fields(b, s: int, e: int):
    """(field number, wire type, value) of a message in b[s:e]; LEN values are (start, end)."""
    p = s
    while p < e:
        key, p = _varint(b, p)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, p = _varint(b, p)
        elif wt == 1:
            v, p = (p, p + 8), p + 8
        elif wt == 2:
            n, p = _varint(b, p)
            v, p = (p, p + n), p + n
        elif wt == 5:
            v, p = (p, p + 4), p + 4
        else:
            raise ValueError(f'unsupported protobuf wire type {wt}')
        if p > e:
            raise ValueError('a field runs past the end of its message')
        yield fn, wt, v


def _int64(v: int) -> int:
    return v - (1 << 64) if v >= 1 << 63 else v


def _feature(b, s, e):
    """A tf.train.Feature -> ('bytes', [bytes]) | ('float', float32 array) | ('int64', int64 array) | ('none', [])."""
    kind, vals = 'none', []
    for fn, wt, v in _fields(b, s, e):
        if wt != 2 or fn not in (1, 2, 3):
            continue
        if fn == 1:
            kind, vals = 'bytes', [bytes(b[x:y]) for f2, w2, (x, y) in _fields(b, *v) if f2 == 1 and w2 == 2]
        elif fn == 2:
            kind, parts = 'float', []
            for f2, w2, (x, y) in _fields(b, *v):
                if f2 == 1 and w2 in (2, 5):
                    parts.append(np.frombuffer(bytes(b[x:y]), '<f4'))
            vals = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        else:
            kind, out = 'int64', []
            for f2, w2, val in _fields(b, *v):
                if f2 != 1:
                    continue
                if w2 == 0:
                    out.append(_int64(val))
                elif w2 == 2:
                    p, end = val
                    while p < end:
                        x, p = _varint(b, p)
                        out.append(_int64(x))
            vals = np.asarray(out, np.int64)
    return kind, vals


def _features(b, s, e) -> dict:
    """A tf.train.Features (map<string, Feature>) -> {name: (kind, values)}."""
    out = {}
    for fn, wt, v in _fields(b, s, e):
        if fn != 1 or wt != 2:
            continue
        key, val = '', ('none', [])
        for f2, w2, (x, y) in _fields(b, *v):
            if f2 == 1 and w2 == 2:
                key = bytes(b[x:y]).decode('utf-8', 'replace')
            elif f2 == 2 and w2 == 2:
                val = _feature(b, x, y)
        out[key] = val
    return out


def parse_example(data) -> dict:
    """Serialized tf.train.Example -> {feature name: (kind, values)}."""
    b = memoryview(data)
    out = {}
    try:
        for fn, wt, v in _fields(b, 0, len(b)):
            if fn == 1 and wt == 2:
                out.update(_features(b, *v))
    except (ValueError, IndexError) as e:
        raise DatasetError(f'A record is not a readable tf.train.Example ({e}).') from None
    return out


def parse_sequence_example(data) -> tuple:
    """Serialized tf.train.SequenceExample -> (context {name: (kind, values)}, {name: [(kind, values) per step]})."""
    b = memoryview(data)
    ctx, lists = {}, {}
    try:
        for fn, wt, v in _fields(b, 0, len(b)):
            if wt != 2:
                continue
            if fn == 1:
                ctx.update(_features(b, *v))
            elif fn == 2:
                for f2, w2, ent in _fields(b, *v):          # FeatureLists: map<string, FeatureList> feature_list = 1
                    if f2 != 1 or w2 != 2:
                        continue
                    key, steps = '', []
                    for f3, w3, (x, y) in _fields(b, *ent):
                        if f3 == 1 and w3 == 2:
                            key = bytes(b[x:y]).decode('utf-8', 'replace')
                        elif f3 == 2 and w3 == 2:           # FeatureList: repeated Feature feature = 1
                            steps = [_feature(b, x2, y2) for f4, w4, (x2, y2) in _fields(b, x, y)
                                     if f4 == 1 and w4 == 2]
                    lists[key] = steps
    except (ValueError, IndexError) as e:
        raise DatasetError(f'A record is not a readable tf.train.SequenceExample ({e}).') from None
    return ctx, lists


def is_sequence_example(data) -> bool:
    """True when the record has a feature_lists field (field 2): a SequenceExample, not an Example."""
    b = memoryview(data)
    try:
        return any(fn == 2 for fn, _, _ in _fields(b, 0, len(b)))
    except (ValueError, IndexError):
        return False


# ---------------------------------------------------------------------------------------------------------------------
# TFDS feature specs

_DT = {'bool': np.bool_, 'float32': np.float32, 'float64': np.float64, 'float16': np.float16, 'int8': np.int8,
       'int16': np.int16, 'int32': np.int32, 'int64': np.int64, 'uint8': np.uint8, 'uint16': np.uint16,
       'uint32': np.uint32, 'uint64': np.uint64}


def flatten_features(doc: dict) -> dict:
    """features.json -> {flat name: {'cls', 'shape' [ints, -1 dynamic], 'dtype', 'encoding', 'image_format',
    'in_steps', 'description'}} in the file's order. Names are joined with '/' as TFDS flattens them."""
    out = {}

    def leaf_shape(d):
        dims = (d.get('shape') or {}).get('dimensions') or []
        return [int(x) for x in dims]

    def walk(f, path, in_steps):
        if not isinstance(f, dict):
            return
        cls = str(f.get('pythonClassName', '')).split('.')[-1]
        if 'featuresDict' in f:
            for k, v in ((f['featuresDict'] or {}).get('features') or {}).items():
                walk(v, path + [k], in_steps)
        elif 'sequence' in f:
            seq = f['sequence'] or {}
            if in_steps:
                out['/'.join(path)] = {'cls': 'nested_sequence', 'shape': [], 'dtype': '', 'encoding': '',
                                       'image_format': None, 'in_steps': True, 'description': ''}
                return
            walk(seq.get('feature') or {}, path, True)
        elif 'image' in f:
            d = f['image'] or {}
            out['/'.join(path)] = {'cls': cls or 'Image', 'shape': leaf_shape(d), 'dtype': d.get('dtype', 'uint8'),
                                   'encoding': 'image', 'image_format': d.get('encodingFormat'),
                                   'in_steps': in_steps, 'description': f.get('description', '')}
        else:
            d = f.get('tensor') or f.get('text') or {}
            dtype = d.get('dtype', 'string' if 'text' in f else '')
            out['/'.join(path)] = {'cls': cls, 'shape': leaf_shape(d), 'dtype': dtype,
                                   'encoding': str(d.get('encoding', 'none')).lower(), 'image_format': None,
                                   'in_steps': in_steps, 'description': f.get('description', '')}
    walk(doc, [], False)
    return out


def _is_image_bytes(b: bytes) -> str | None:
    if b[:3] == b'\xff\xd8\xff':
        return 'jpeg'
    if b[:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'
    return None


def _decode_values(kind, vals, spec, n_steps):
    """One flat feature -> (array of N rows | list of N bytes for images/strings, note). spec may be None."""
    shape = list(spec['shape']) if spec else None
    dtype = spec['dtype'] if spec else ''
    enc = spec['encoding'] if spec else 'none'
    if spec and spec['encoding'] == 'image':
        return list(vals), None
    if kind == 'bytes':
        if enc in ('bytes', 'zlib') and dtype in _DT:
            dt = np.dtype(_DT[dtype]).newbyteorder('<')
            rows = []
            for v in vals:
                raw = zlib.decompress(v) if enc == 'zlib' else v
                a = np.frombuffer(raw, dt)
                rows.append(a.reshape(shape) if shape and -1 not in shape and a.size == int(np.prod(shape)) else a)
            try:
                return np.stack(rows) if rows else np.zeros([0] + (shape or [])), None
            except ValueError:
                return rows, 'rows of different shapes'
        return list(vals), None
    a = np.asarray(vals)
    if kind == 'int64' and dtype == 'uint64':
        a = a.view(np.uint64)
    if dtype in _DT:
        a = a.astype(_DT[dtype]) if dtype != 'bool' else a != 0
    if n_steps is None:
        return a, None
    per = int(np.prod(shape)) if shape and -1 not in shape else (a.size // n_steps if n_steps else 0)
    if n_steps and per and a.size == per * n_steps:
        return a.reshape([n_steps] + (shape if shape and -1 not in shape else [per])), None
    if n_steps == 0:
        return a.reshape([0] + (shape or [])) if shape and -1 not in shape else a, None
    return a, f'{a.size} values do not divide into {n_steps} steps'


# ---------------------------------------------------------------------------------------------------------------------
# the dataset

STATE_NAMES = ('joint_position', 'joint_positions', 'joint_pos', 'joint_state', 'joint_states', 'qpos', 'state',
               'robot_state', 'proprio')
TASK_NAMES = ('language_instruction', 'natural_language_instruction', 'instruction', 'task')
_SHARD_RE = re.compile(r'^(?P<name>.+?)-(?P<split>[^-.]+)\.tfrecord(?:-(?P<x>\d+)-of-(?P<y>\d+))?(?:\.gz)?$')
_VERSION_RE = re.compile(r'^\d+(\.\d+)*$')


def _version_key(v):
    return tuple(int(x) for x in v.split('.'))


def _root(path: str) -> str | None:
    """The folder (or file) holding the TFRecords: path itself, or its highest version folder (TFDS's
    <name>/<version>/ layout)."""
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        return None
    if glob.glob(os.path.join(glob.escape(path), '*.tfrecord*')):
        return path
    vers = sorted((d for d in os.listdir(path) if _VERSION_RE.match(d) and os.path.isdir(os.path.join(path, d))),
                  key=_version_key)
    for v in reversed(vers):
        sub = os.path.join(path, v)
        if glob.glob(os.path.join(glob.escape(sub), '*.tfrecord*')):
            return sub
    return None


def _first_record_ok(path: str) -> bool:
    with open(path, 'rb') as f:
        head = f.read(12)
        if head[:2] == b'\x1f\x8b':
            with gzip.open(path, 'rb') as g:
                head = g.read(12)
    if len(head) < 12:
        return False
    n, mcrc = struct.unpack('<QI', head)
    return mask_crc(crc32c(head[:8])) == mcrc


def sniff(path: str) -> bool:
    if os.path.isfile(path):
        return ('.tfrecord' in os.path.basename(path) or path.endswith('.tfrec')) and _first_record_ok(path)
    root = _root(path)
    if root is None:
        return False
    files = sorted(glob.glob(os.path.join(glob.escape(root), '*.tfrecord*')))
    return bool(files) and _first_record_ok(files[0])


def _load_json(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


class RldsDataset(Dataset):
    """An RLDS / Open X-Embodiment dataset in TFDS's on-disk form (a folder of <name>-<split>.tfrecord-XXXXX-of-YYYYY
    shards with dataset_info.json and features.json), or a single TFRecord file. fps: the control rate to time the
    steps by (a user setting; None leaves t counting steps)."""

    format = 'rlds'

    def __init__(self, path: str, fps: float | None = None):
        root = _root(path)
        if root is None:
            raise DatasetError(f'{os.path.basename(path) or path} holds no TFRecord files.')
        self.path = os.path.abspath(path)
        self.root = root
        self.fps = float(fps) if fps else None
        folder = root if os.path.isdir(root) else os.path.dirname(root)
        self.info = _load_json(os.path.join(folder, 'dataset_info.json')) or {}
        fdoc = _load_json(os.path.join(folder, 'features.json'))
        self.features = flatten_features(fdoc) if isinstance(fdoc, dict) else None
        self.version = str(self.info.get('version', ''))
        self.name = str(self.info.get('name') or os.path.basename(os.path.normpath(folder)))
        files = [root] if os.path.isfile(root) else sorted(glob.glob(os.path.join(glob.escape(root), '*.tfrecord*')))
        split_order = [s.get('name') for s in self.info.get('splits', []) if isinstance(s, dict)]
        shards = []
        for fp in files:
            m = _SHARD_RE.match(os.path.basename(fp))
            split = m.group('split') if m else ''
            x = int(m.group('x')) if m and m.group('x') else 0
            rank = split_order.index(split) if split in split_order else len(split_order)
            shards.append((rank, split, x, fp))
        shards.sort()
        self._shards = [(split, fp) for _, split, _, fp in shards]
        self._files = {}
        self._refs = None
        self._folder = folder

    def _file(self, fp) -> TFRecordFile:
        if fp not in self._files:
            self._files[fp] = TFRecordFile(fp)
        return self._files[fp]

    def episodes(self) -> list:
        if self._refs is None:
            refs, per_split = [], {}
            for split, fp in self._shards:
                for k, (off, n) in enumerate(self._file(fp).index()):
                    j = per_split.get(split, 0)
                    per_split[split] = j + 1
                    eid = f'{split}-{j:06d}' if split else f'{os.path.basename(fp)}-{k:06d}'
                    refs.append(EpisodeRef(len(refs), eid, None, {'split': split, 'file': os.path.basename(fp),
                                                                  'record': k, 'offset': off, 'bytes': n}))
            self._refs = refs
        return self._refs

    def truth(self):
        doc = _load_json(os.path.join(self._folder, 'dexvantage_truth.json'))
        return doc if isinstance(doc, dict) else None

    def load(self, index: int) -> Episode:
        refs = self.episodes()
        if not 0 <= index < len(refs):
            raise DatasetError(f'The dataset has {len(refs)} episodes; there is no episode {index + 1}.')
        ref = refs[index]
        fp = next(f for s, f in self._shards if os.path.basename(f) == ref.meta['file'])
        data = self._file(fp).read(ref.meta['offset'], ref.meta['record'])
        if is_sequence_example(data):
            ctx, lists = parse_sequence_example(data)
            steps, n, extra = self._steps_from_lists(lists)
            episode_level = dict(ctx, **extra)
        else:
            flat = parse_example(data)
            steps, n, episode_level = self._steps_from_example(flat)
        return self._episode(ref, steps, n, episode_level)

    # --- decoding

    def _spec(self, name):
        if self.features is None:
            return None
        return self.features.get(name) or self.features.get('steps/' + name)

    def _steps_from_example(self, flat: dict):
        """{step feature (without 'steps/'): values}, N, {episode-level feature: values}."""
        step_keys = [k for k in flat if k.startswith('steps/')]
        if self.features is not None:
            step_keys = [k for k in flat if (self.features.get(k) or {}).get('in_steps')] or step_keys
        n = None
        for probe in ('steps/is_first', 'steps/is_last'):
            if probe in flat and flat[probe][0] == 'int64':
                n = len(flat[probe][1])
                break
        if n is None:
            for k in step_keys:
                spec = self._spec(k)
                if flat[k][0] == 'bytes' and (spec is None or spec['encoding'] in ('image', 'bytes', 'zlib')
                                              or spec['dtype'] == 'string'):
                    n = len(flat[k][1])
                    break
        if n is None:
            for k in step_keys:
                spec = self._spec(k)
                if spec and spec['shape'] is not None and -1 not in spec['shape']:
                    n = len(flat[k][1]) // max(1, int(np.prod(spec['shape'])))
                    break
        if n is None:
            n = len(flat[step_keys[0]][1]) if step_keys else 0
        steps, notes = {}, []
        for k in step_keys:
            spec = self._spec(k)
            if spec and spec['cls'] == 'nested_sequence':
                notes.append(f'{k}: a nested sequence, left out')
                continue
            if any(k.endswith(s) for s in ('/ragged_flat_values',)) or '/ragged_row_lengths_' in k:
                notes.append(f'{k}: a ragged feature, left out')
                continue
            vals, note = _decode_values(*flat[k], spec, n)
            if note:
                notes.append(f'{k}: {note}')
            steps[k[len('steps/'):]] = (vals, spec)
        episode_level = {k: v for k, v in flat.items() if k not in step_keys}
        if notes:
            episode_level['__notes__'] = ('notes', notes)
        return steps, n, episode_level

    def _steps_from_lists(self, lists: dict):
        n = max((len(v) for v in lists.values()), default=0)
        steps, notes = {}, []
        for k, per_step in lists.items():
            name = k[len('steps/'):] if k.startswith('steps/') else k
            spec = self._spec(k) or self._spec(name)
            if len(per_step) != n:
                notes.append(f'{k}: {len(per_step)} steps, not {n}: left out')
                continue
            kinds = {p[0] for p in per_step}
            kind = kinds.pop() if len(kinds) == 1 else 'mixed'
            if kind == 'bytes':
                if spec is None or spec['encoding'] == 'image' or spec['dtype'] == 'string':
                    vals = [p[1][0] if p[1] else b'' for p in per_step]
                else:
                    vals, note = _decode_values('bytes', [p[1][0] for p in per_step], spec, n)
                    if note:
                        notes.append(f'{k}: {note}')
            elif kind in ('float', 'int64'):
                vals, note = _decode_values(kind, np.concatenate([np.asarray(p[1]) for p in per_step]) if per_step
                                            else np.zeros(0), spec, n)
                if note:
                    notes.append(f'{k}: {note}')
            else:
                notes.append(f'{k}: features of mixed kinds, left out')
                continue
            steps[name] = (vals, spec)
        return steps, n, {'__notes__': ('notes', notes)} if notes else {}

    def _episode(self, ref, steps: dict, n: int, episode_level: dict) -> Episode:
        meta = {'split': ref.meta['split'], 'file': ref.meta['file'], 'record': ref.meta['record'],
                'dataset': self.name, 'warnings': []}
        notes = episode_level.pop('__notes__', None)
        if notes:
            meta['warnings'] += list(notes[1])

        def numeric(name):
            v = steps.get(name)
            if v is None or not isinstance(v[0], np.ndarray) or v[0].dtype.kind not in 'biuf':
                return None
            a = v[0]
            if n == 0:
                return np.zeros((0, int(np.prod(a.shape[1:])) if a.ndim > 1 else 1), a.dtype)
            return a.reshape(len(a), -1) if a.ndim >= 1 and len(a) == n else None

        def strings(name):
            v = steps.get(name)
            if v is None or not isinstance(v[0], list):
                return None
            spec = v[1]
            if spec is not None and spec['encoding'] == 'image':
                return None
            if v[0] and isinstance(v[0][0], bytes) and _is_image_bytes(v[0][0]):
                return None
            return [x.decode('utf-8', 'replace') if isinstance(x, bytes) else str(x) for x in v[0]]

        # state
        q, joint_names, state_key = np.zeros((n, 0)), [], None
        obs = [k for k in steps if k.startswith('observation/')]
        for want in STATE_NAMES:
            hit = next((k for k in obs if k.split('/')[-1] == want and numeric(k) is not None), None)
            if hit:
                state_key = hit
                break
        if state_key:
            q = numeric(state_key).astype(np.float64)
            base = state_key.split('/')[-1]
            joint_names = [f'{base}_{i}' for i in range(q.shape[1])]
            meta['state_feature'] = state_key
        else:
            meta['warnings'].append('no state vector among the observation features')
        # action
        action, action_names = None, []
        if numeric('action') is not None:
            action = numeric('action').astype(np.float64)
            action_names = [f'action_{i}' for i in range(action.shape[1])]
        else:
            subs = [k for k in steps if k.startswith('action/') and numeric(k) is not None]
            order = [k[len('steps/'):] for k in self.features] if self.features else []
            subs.sort(key=lambda k: (order.index(k), k) if k in order else (len(order), k))
            if subs:
                cols = []
                for k in subs:
                    a = numeric(k).astype(np.float64)
                    cols.append(a)
                    nm = k[len('action/'):]
                    action_names += [nm] if a.shape[1] == 1 else [f'{nm}_{i}' for i in range(a.shape[1])]
                action = np.concatenate(cols, axis=1)
        # time
        t, time_source = None, None
        for k in [k for k in steps if k.split('/')[-1] == 'timestamp']:
            a = numeric(k)
            if a is not None and a.shape[1] == 1 and a.dtype.kind == 'f':
                t = a[:, 0].astype(np.float64)
                t = t - t[0] if n else t
                time_source = f'feature {k} (read as seconds)'
                break
        if t is None:
            if self.fps:
                t = np.arange(n, dtype=np.float64) / self.fps
                time_source = f'step index / {self.fps:g} fps (the fps given to the reader)'
            else:
                t = np.arange(n, dtype=np.float64)
                time_source = 'step index (the dataset records no times and no fps was given)'
                meta['time_base'] = 'steps'
        meta['time_source'] = time_source
        # task
        task = ''
        for k in steps:
            if k.split('/')[-1] in TASK_NAMES:
                s = strings(k)
                if s:
                    task = next((x for x in s if x), '')
                    if task:
                        break
        # RLDS step flags and scalars
        rl = {}
        for k in ('is_first', 'is_last', 'is_terminal', 'reward', 'discount'):
            a = numeric(k)
            if a is not None:
                rl[k] = a[:, 0] if a.shape[1] == 1 else a
        meta['rlds'] = rl
        if 'is_first' in rl and n and (not bool(rl['is_first'][0]) or int(np.sum(rl['is_first'])) != 1):
            meta['warnings'].append('is_first is not set on exactly the first step')
        if 'is_last' in rl and n and (not bool(rl['is_last'][-1]) or int(np.sum(rl['is_last'])) != 1):
            meta['warnings'].append('is_last is not set on exactly the last step')
        # episode metadata
        em = {}
        for k, (kind, vals) in episode_level.items():
            key = k[len('episode_metadata/'):] if k.startswith('episode_metadata/') else k
            if kind == 'bytes':
                vv = [v.decode('utf-8', 'replace') if not _is_image_bytes(v) else f'<{len(v)} bytes image>'
                      for v in vals]
                em[key] = vv[0] if len(vv) == 1 else vv
            elif kind in ('float', 'int64'):
                a = np.asarray(vals)
                em[key] = a.tolist() if a.size != 1 else a.reshape(-1)[0].item()
        meta['episode_metadata'] = em
        if not task:
            task = next((str(em[k]) for k in TASK_NAMES if isinstance(em.get(k), str) and em.get(k)), '')
        # cameras
        cams, depth = {}, {}
        for k, (vals, spec) in steps.items():
            if not isinstance(vals, list) or not vals or not isinstance(vals[0], (bytes, bytearray)):
                continue
            fmt = (spec or {}).get('image_format') or _is_image_bytes(vals[0])
            if not ((spec and spec['encoding'] == 'image') or _is_image_bytes(vals[0])):
                continue
            shp = (spec or {}).get('shape') or []
            dtype = (spec or {}).get('dtype') or 'uint8'
            if dtype != 'uint8' or (len(shp) == 3 and shp[2] not in (1, 3, 4, -1)):
                depth[k] = {'count': len(vals), 'dtype': dtype, 'shape': shp, 'encoding': fmt}
                continue
            h, w = (shp[0], shp[1]) if len(shp) >= 2 and shp[0] > 0 and shp[1] > 0 else (None, None)
            if h is None:
                first = _decode_image(vals[0], k)
                h, w = first.shape[:2]
            cams[k] = CameraStream(key=k, width=int(w), height=int(h), fps=self.fps, timestamps=t.copy(),
                                   meta={'encoding': fmt, 'feature': 'steps/' + k, 'source': 'rlds'},
                                   _decode=_image_decoder(vals, k))
        if depth:
            meta['depth_streams'] = depth
        return Episode(id=ref.id, index=ref.index, t=t, q=q, joint_names=joint_names, action=action,
                       action_names=action_names, fps=self.fps, robot_type='', task=task, cameras=cams, meta=meta)


def _decode_image(b: bytes, key: str) -> np.ndarray:
    import cv2
    img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise DatasetError(f'A frame of camera {key} is not a readable JPEG or PNG image.')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _image_decoder(vals: list, key: str):
    def decode(indices):
        idx = range(len(vals)) if indices is None else indices
        for i in idx:
            if not 0 <= int(i) < len(vals):
                raise DatasetError(f'Camera {key} has {len(vals)} frames; there is no frame {int(i)}.')
            yield int(i), _decode_image(vals[int(i)], key)
    return decode


@register_reader('rlds', sniff)
def open_rlds(path: str, fps: float | None = None) -> RldsDataset:
    """Open an RLDS / Open X-Embodiment dataset folder (or one TFRecord file)."""
    return RldsDataset(path, fps=fps)
