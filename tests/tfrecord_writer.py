"""A TFRecord and tf.train.Example writer for the RLDS reader's tests, written from the format documents and kept
apart from the reader's code: a bit-at-a-time CRC-32C (no table, no numpy), a protobuf encoder from the wire-format
guide (https://protobuf.dev/programming-guides/encoding/), TFDS's flattening of nested features ('steps/observation/
image', a sequence stacked and stored flat) and features.json / dataset_info.json in the form TFDS writes them
(compared with the published features.json of the Open X-Embodiment datasets on 2026-11-07)."""
from __future__ import annotations

import gzip
import json
import os
import struct
import zlib

import numpy as np


def crc32c_bitwise(data: bytes) -> int:
    c = 0xFFFFFFFF
    for b in data:
        c ^= b
        for _ in range(8):
            c = (c >> 1) ^ (0x82F63B78 if c & 1 else 0)
    return c ^ 0xFFFFFFFF


def masked(c: int) -> int:
    return ((((c >> 15) | (c << 17)) & 0xFFFFFFFF) + 0xA282EAD8) & 0xFFFFFFFF


def frame(data: bytes) -> bytes:
    head = struct.pack('<Q', len(data))
    return (head + struct.pack('<I', masked(crc32c_bitwise(head))) + data
            + struct.pack('<I', masked(crc32c_bitwise(data))))


def write_tfrecord(path: str, records, compress: bool = False) -> None:
    blob = b''.join(frame(r) for r in records)
    if compress:
        with gzip.open(path, 'wb') as f:
            f.write(blob)
    else:
        with open(path, 'wb') as f:
            f.write(blob)


# --- protobuf

def varint(v: int) -> bytes:
    if v < 0:
        v += 1 << 64
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def ld(field: int, payload: bytes) -> bytes:
    return varint(field << 3 | 2) + varint(len(payload)) + payload


def feature(kind: str, values, packed: bool = True) -> bytes:
    """A tf.train.Feature: kind 'bytes' | 'float' | 'int64'."""
    if kind == 'bytes':
        inner = b''.join(ld(1, v if isinstance(v, bytes) else str(v).encode()) for v in values)
        return ld(1, inner)
    if kind == 'float':
        vals = [float(v) for v in values]
        if packed:
            inner = ld(1, struct.pack(f'<{len(vals)}f', *vals)) if vals else b''
        else:
            inner = b''.join(varint(1 << 3 | 5) + struct.pack('<f', v) for v in vals)
        return ld(2, inner)
    vals = [int(v) for v in values]
    if packed:
        inner = ld(1, b''.join(varint(v) for v in vals)) if vals else b''
    else:
        inner = b''.join(varint(1 << 3 | 0) + varint(v) for v in vals)
    return ld(3, inner)


def features(d: dict) -> bytes:
    """tf.train.Features of {name: encoded Feature bytes}: map entries key = 1, value = 2."""
    return b''.join(ld(1, ld(1, k.encode()) + ld(2, v)) for k, v in d.items())


def example(d: dict) -> bytes:
    return ld(1, features(d))


def sequence_example(context: dict, lists: dict) -> bytes:
    """context {name: Feature bytes}; lists {name: [Feature bytes per step]}."""
    fl = b''.join(ld(1, ld(1, k.encode()) + ld(2, b''.join(ld(1, f) for f in steps))) for k, steps in lists.items())
    return ld(1, features(context)) + ld(2, fl)


# --- TFDS on-disk form

def _tensor(shape, dtype, encoding='none', cls='tensor_feature.Tensor', description=None):
    d = {'pythonClassName': f'tensorflow_datasets.core.features.{cls}',
         'tensor': {'shape': {'dimensions': [str(s) for s in shape]} if shape else {}, 'dtype': dtype,
                    'encoding': encoding}}
    if description:
        d['description'] = description
    return d


def _image(shape, fmt, dtype='uint8'):
    d = {'pythonClassName': 'tensorflow_datasets.core.features.image_feature.Image',
         'image': {'shape': {'dimensions': [str(s) for s in shape]}, 'dtype': dtype}}
    if fmt:
        d['image']['encodingFormat'] = fmt
    return d


def _dict(features):
    return {'pythonClassName': 'tensorflow_datasets.core.features.features_dict.FeaturesDict',
            'featuresDict': {'features': features}}


def oxe_features(h, w, state_dim=7, zlib_extra=False):
    obs = {'image': _image([h, w, 3], 'jpeg'), 'wrist_image': _image([h, w, 3], 'png'),
           'depth': _image([h, w, 1], 'png', 'float32'),
           'state': _tensor([state_dim], 'float32', description='joint angles and gripper'),
           'natural_language_instruction': _tensor([], 'string')}
    if zlib_extra:
        obs['big'] = _tensor([4, 2], 'float64', encoding='zlib')
        obs['raw'] = _tensor([3], 'int32', encoding='bytes')
    steps = _dict({'is_first': _tensor([], 'bool'), 'is_last': _tensor([], 'bool'),
                   'is_terminal': _tensor([], 'bool'), 'reward': _tensor([], 'float32', cls='scalar.Scalar'),
                   'observation': _dict(obs),
                   'action': _dict({'world_vector': _tensor([3], 'float32'),
                                    'rotation_delta': _tensor([3], 'float32'),
                                    'open_gripper': _tensor([], 'bool')})})
    return _dict({'steps': {'pythonClassName': 'tensorflow_datasets.core.features.dataset_feature.Dataset',
                            'sequence': {'feature': steps, 'length': '-1'}},
                  'episode_metadata': _dict({'file_path': _tensor([], 'string')})})


def make_episode(rng, n, h, w, state_dim=7, zlib_extra=False):
    """(feature dict for example(), the values sent) of one TFDS-style RLDS episode."""
    import cv2
    imgs = rng.integers(0, 256, (n, h, w, 3), dtype=np.uint8)
    smooth = np.zeros((n, h, w, 3), np.uint8)
    for i in range(n):
        smooth[i] = np.linspace(20 + 5 * i, 200, w, dtype=np.uint8)[None, :, None]
    jpgs = [cv2.imencode('.jpg', cv2.cvtColor(smooth[i], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])[1]
            .tobytes() for i in range(n)]
    pngs = [cv2.imencode('.png', cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR))[1].tobytes() for i in range(n)]
    dep = rng.uniform(0.2, 2.0, (n, h, w)).astype(np.float32)
    dpng = [cv2.imencode('.png', dep[i].view(np.uint8).reshape(h, w, 4))[1].tobytes() for i in range(n)]
    state = rng.normal(size=(n, state_dim)).astype(np.float32)
    wv = rng.normal(size=(n, 3)).astype(np.float32)
    rd = rng.normal(size=(n, 3)).astype(np.float32)
    og = rng.random(n) < 0.5
    first = np.zeros(n, bool)
    first[0] = True
    last = np.zeros(n, bool)
    last[-1] = True
    reward = np.zeros(n, np.float32)
    reward[-1] = 1.0
    instr = 'put the red block in the bowl'
    d = {'steps/is_first': feature('int64', first.astype(int)), 'steps/is_last': feature('int64', last.astype(int)),
         'steps/is_terminal': feature('int64', last.astype(int)), 'steps/reward': feature('float', reward),
         'steps/observation/image': feature('bytes', jpgs), 'steps/observation/wrist_image': feature('bytes', pngs),
         'steps/observation/depth': feature('bytes', dpng),
         'steps/observation/state': feature('float', state.reshape(-1)),
         'steps/observation/natural_language_instruction': feature('bytes', [instr.encode()] * n),
         'steps/action/world_vector': feature('float', wv.reshape(-1)),
         'steps/action/rotation_delta': feature('float', rd.reshape(-1), packed=False),
         'steps/action/open_gripper': feature('int64', og.astype(int), packed=False),
         'episode_metadata/file_path': feature('bytes', [b'episode_raw/ep.npy'])}
    sent = {'smooth': smooth, 'imgs': imgs, 'state': state, 'world_vector': wv, 'rotation_delta': rd,
            'open_gripper': og, 'task': instr, 'n': n}
    if zlib_extra:
        big = rng.normal(size=(n, 4, 2))
        raw = rng.integers(-1000, 1000, (n, 3)).astype(np.int32)
        d['steps/observation/big'] = feature('bytes', [zlib.compress(big[i].astype('<f8').tobytes())
                                                       for i in range(n)])
        d['steps/observation/raw'] = feature('bytes', [raw[i].astype('<i4').tobytes() for i in range(n)])
        sent.update(big=big, raw=raw)
    return d, sent


def write_oxe(root, name, splits, h=24, w=32, seed=0, zlib_extra=False):
    """A TFDS dataset folder <root>/<name>/0.1.0 with splits {split: [episode lengths per shard]}; returns
    (version folder, {split: [sent episode values in order]})."""
    rng = np.random.default_rng(seed)
    folder = os.path.join(root, name, '0.1.0')
    os.makedirs(folder, exist_ok=True)
    sent, split_docs = {}, []
    for split, shards in splits.items():
        sent[split] = []
        for k, lengths in enumerate(shards):
            recs = []
            for n in lengths:
                d, s = make_episode(rng, n, h, w, zlib_extra=zlib_extra)
                recs.append(example(d))
                sent[split].append(s)
            write_tfrecord(os.path.join(folder, f'{name}-{split}.tfrecord-{k:05d}-of-{len(shards):05d}'), recs)
        split_docs.append({'filepathTemplate': '{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}', 'name': split,
                           'shardLengths': [str(len(s)) for s in shards]})
    with open(os.path.join(folder, 'features.json'), 'w') as f:
        json.dump(oxe_features(h, w, zlib_extra=zlib_extra), f, indent=4)
    with open(os.path.join(folder, 'dataset_info.json'), 'w') as f:
        json.dump({'name': name, 'version': '0.1.0', 'fileFormat': 'tfrecord', 'moduleName': 'test',
                   'splits': split_docs}, f, indent=2)
    return folder, sent
