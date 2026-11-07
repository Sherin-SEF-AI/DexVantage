"""ROS 2 bags (MCAP and SQLite .db3) as datasets: one episode per bag, read through core/rosbag.py (ported from Vantage
core/rosbag.py) with no ROS installation.

What a path may be: a rosbag2 bag (a folder with metadata.yaml and its .mcap / .db3 files), a single .mcap or .db3
file, or a folder of bags (sub-folders that are bags, and .mcap / .db3 files lying directly in it), each one episode.

Mapping onto Episode (episode.py):

- joint state: the sensor_msgs/JointState topic '/joint_states' (else the JointState topic with the most messages;
  the others are listed in meta). t is each message's header stamp, or its log (receive) time when the driver leaves
  the stamp at zero; q holds the union of the joint names in first-seen order. A message that leaves a joint out holds
  that joint's previous value (counted in meta['q_held']); a joint not yet seen is NaN.
- cameras: every sensor_msgs/Image and sensor_msgs/CompressedImage topic with 8-bit colour or mono pixels is a
  CameraStream keyed by its topic; timestamps are the header stamps, the log times are in meta['log_times']. Depth and
  16-bit / float images are listed in episode.meta['depth_streams'] (not decoded). When a camera's header stamps run on
  another clock than the joint states (their offset from the log time differs from the joint states' by more than
  CLOCK_SPLIT_S, a labelled default of 10 s: a device clock since power-on, not the host's), the header stamps become
  hw_timestamps and timestamps are the log times carried onto the joint-state clock. A sensor_msgs/TimeReference topic
  in the camera's namespace (header.stamp = the frame's stamp, time_ref = the device time) also gives hw_timestamps.
  hw_timestamps are seconds from that camera's first device time (meta['hw_t0_s'] keeps the raw first value).
- intrinsics: the first sensor_msgs/CameraInfo in the camera's namespace (<ns>/camera_info beside <ns>/image_raw, or
  the same header.frame_id); plumb_bob and rational_polynomial map to the 'brown' lens model ('pinhole' when every
  coefficient is zero), equidistant to 'kb'; a CameraInfo whose K is all zeros (uncalibrated) gives None.
- extrinsics: the static TF tree (/tf_static): the pose of the image's header.frame_id in the top frame reachable
  through static transforms (the world or base frame of a fixed camera; the link a wrist camera is bolted to), with
  the chain of frames in meta['tf_chain'].
- task: a std_msgs/String topic named task, instruction, language_instruction or prompt (first message), else the
  metadata.yaml custom_data entry of that name; robot_type: the robot name of a /robot_description URDF.
- times: seconds from the episode start, the earliest joint-state or camera time, computed in integer nanoseconds
  before the conversion so a 2026 epoch time keeps nanosecond resolution.

Known limits: one robot state vector (from one JointState topic); actions are not read from bags (there is no
standard action message; command topics are listed in meta); /tf (dynamic) is not used for extrinsics; vendor frame
metadata topics (exposure, gain) are not decoded; a header.frame_id that is the camera's body frame rather than its
optical frame gives that frame's pose (meta['frame_id'] says which); Bayer and YUV 4:2:2 pixels are converted with
OpenCV's demosaicing; without a JointState topic there is no reference clock and each camera keeps its
header stamps; rosbag2 'file' compression (a whole .mcap.zstd) is refused with the command that undoes it.
"""
from __future__ import annotations

import os
import re
import struct

import numpy as np

from .. import rosbag as RB
from .episode import CameraStream, Dataset, DatasetError, Episode, EpisodeRef, register_reader

CLOCK_SPLIT_S = 10.0           # default: header-vs-log offsets further apart than this are different clocks
_BAG_EXT = ('.mcap', '.db3')
_IMAGE = 'sensor_msgs/msg/Image'
_CIMAGE = 'sensor_msgs/msg/CompressedImage'
_INFO = 'sensor_msgs/msg/CameraInfo'
_JOINTS = 'sensor_msgs/msg/JointState'
_TF = 'tf2_msgs/msg/TFMessage'
_TIMEREF = 'sensor_msgs/msg/TimeReference'
_STRING = 'std_msgs/msg/String'
TASK_TOPICS = ('task', 'instruction', 'language_instruction', 'prompt')
_COLOUR = {'rgb8', 'bgr8', 'rgba8', 'bgra8', 'mono8', 'rgb16', 'bgr16', 'rgba16', 'bgra16', '8UC1', '8UC3', '8UC4',
           'yuv422', 'uyvy', 'yuyv', 'yuv422_yuy2'} | {f'bayer_{p}8' for p in ('rggb', 'bggr', 'gbrg', 'grbg')}


def _magic(path):
    try:
        with open(path, 'rb') as f:
            m = f.read(16)
    except OSError:
        return ''
    if m.startswith(RB.MCAP_MAGIC):
        return 'mcap'
    if m.startswith(RB._SQLITE_MAGIC):
        return 'sqlite3'
    return ''


def _metadata(path):
    meta = os.path.join(path, 'metadata.yaml')
    if not os.path.isfile(meta):
        return None
    import yaml
    try:
        with open(meta, encoding='utf-8') as f:
            doc = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return None
    info = doc.get('rosbag2_bagfile_information') if isinstance(doc, dict) else None
    return info if isinstance(info, dict) else None


def _bag_files(path):
    try:
        names = os.listdir(path)
    except OSError:
        return []
    return sorted((os.path.join(path, n) for n in names if n.endswith(_BAG_EXT)
                   and os.path.isfile(os.path.join(path, n))), key=RB._natural)


def _storage(ep):
    """'mcap' | 'sqlite3' | '' of one episode path."""
    if os.path.isfile(ep):
        return _magic(ep)
    info = _metadata(ep)
    if info and info.get('storage_identifier'):
        return 'sqlite3' if 'sqlite' in str(info['storage_identifier']) else str(info['storage_identifier'])
    files = _bag_files(ep)
    kind = _magic(files[0]) if files else ''
    if not kind and files and info is not None:   # a bag's metadata.yaml beside a damaged file: the opener says why
        kind = 'sqlite3' if files[0].endswith('.db3') else 'mcap'
    return kind


def episode_paths(path: str) -> list:
    """The bags under `path`, one per episode, in natural name order (see the module docstring)."""
    if os.path.isfile(path):
        return [path] if _magic(path) else []
    if not os.path.isdir(path):
        return []
    if _metadata(path) is not None:
        return [path]
    out = [f for f in _bag_files(path) if _magic(f)]
    for name in sorted(os.listdir(path), key=lambda n: RB._natural(n)):
        sub = os.path.join(path, name)
        if os.path.isdir(sub) and (_metadata(sub) is not None or any(_magic(f) for f in _bag_files(sub))):
            out.append(sub)
    return sorted(out, key=RB._natural)


def _sniff(kind):
    def sniff(path):
        eps = episode_paths(path)
        return bool(eps) and _storage(eps[0]) == kind
    return sniff


# ---------------------------------------------------------------------------------------------------------------------
# reading a header without decoding a whole image

def _peek(raw, compressed: bool):
    """(stamp ns, frame_id, width, height, encoding or format) of a raw Image / CompressedImage message, reading only
    the fields before the pixel data (width and height None for a compressed image)."""
    mv = memoryview(raw)
    if len(mv) < 16 or mv[0] != 0 or mv[1] > 1:
        raise ValueError('not plain CDR')
    e = '<' if mv[1] == 1 else '>'
    b = mv[4:]
    sec, nsec, n = struct.unpack_from(e + 'iII', b, 0)
    p = 12
    frame = bytes(b[p:p + max(n - 1, 0)]).decode('utf-8', 'replace')
    p += n
    p += -p % 4
    if compressed:
        n2, = struct.unpack_from(e + 'I', b, p)
        fmt = bytes(b[p + 4:p + 4 + max(n2 - 1, 0)]).decode('utf-8', 'replace')
        return sec * 1_000_000_000 + nsec, frame, None, None, fmt
    h, w, n3 = struct.unpack_from(e + 'III', b, p)
    enc = bytes(b[p + 12:p + 12 + max(n3 - 1, 0)]).decode('utf-8', 'replace')
    return sec * 1_000_000_000 + nsec, frame, int(w), int(h), enc


def to_rgb8(img: np.ndarray, encoding: str) -> np.ndarray:
    """An image_array() result in a ROS encoding -> HxWx3 uint8 RGB."""
    import cv2
    enc = encoding.lower()
    a = img
    if enc.startswith('bayer_'):
        code = {'rggb': cv2.COLOR_BayerBG2RGB, 'bggr': cv2.COLOR_BayerRG2RGB, 'gbrg': cv2.COLOR_BayerGR2RGB,
                'grbg': cv2.COLOR_BayerGB2RGB}[enc[6:10]]          # ROS names the pattern, OpenCV the second row
        a = cv2.cvtColor(np.ascontiguousarray(a if a.dtype == np.uint8 else (a >> 8).astype(np.uint8)), code)
        return a
    if enc in ('uyvy', 'yuv422'):
        return cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_YUV2RGB_UYVY)
    if enc in ('yuyv', 'yuv422_yuy2'):
        return cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_YUV2RGB_YUYV)
    if a.dtype == np.uint16:
        a = (a >> 8).astype(np.uint8)
    if a.ndim == 2:
        return np.repeat(a[:, :, None], 3, axis=2)
    if enc.startswith('bgr'):
        a = a[:, :, [2, 1, 0]]
    return np.ascontiguousarray(a[:, :, :3])


def _decoded_rgb(bag, topic, raw, encoding):
    msg = bag.decode(topic, raw)
    img = RB.image_array(msg)
    if 'format' in msg and 'encoding' not in msg:          # compressed: OpenCV decoded it to RGB / RGBA / gray
        if img.dtype != np.uint8:
            img = (img >> 8).astype(np.uint8) if img.dtype == np.uint16 else img.astype(np.uint8)
        return np.repeat(img[:, :, None], 3, axis=2) if img.ndim == 2 else np.ascontiguousarray(img[:, :, :3])
    return to_rgb8(img, msg['encoding'])


def _namespace(topic: str) -> str:
    t = re.sub(r'/(compressed|compressedDepth|theora|ffmpeg|zstd)$', '', topic)
    return t.rsplit('/', 1)[0] if '/' in t.strip('/') else ''


def _lens(info: dict):
    k = [float(v) for v in info.get('k') or []]
    if len(k) != 9 or not any(k[:8]):
        return None
    dm = str(info.get('distortion_model') or '')
    d = [float(v) for v in info.get('d') or []]
    model = {'plumb_bob': 'brown', 'rational_polynomial': 'brown', 'equidistant': 'kb'}.get(dm, dm or 'pinhole')
    if model == 'brown' and not any(d):
        model = 'pinhole'
    return {'model': model, 'width': int(info.get('width') or 0) or None,
            'height': int(info.get('height') or 0) or None, 'K': [k[0:3], k[3:6], k[6:9]], 'dist': d,
            'distortion_model': dm}


def _pose_dict(T: np.ndarray) -> dict:
    from ..mathutil import quat_from_matrix
    return {'position': [float(v) for v in T[:3, 3]], 'quaternion': [float(v) for v in quat_from_matrix(T[:3, :3])]}


def _static_pose(tree: dict, frame: str):
    """(root frame, 4x4 pose of `frame` in it, chain root..frame) through the static tree {child: (parent, T)}."""
    if frame not in tree:
        return None
    T, chain, cur = np.eye(4), [frame], frame
    while cur in tree and len(chain) < 256:
        parent, Tpc = tree[cur]
        T = Tpc @ T
        cur = parent
        chain.append(cur)
    return cur, T, chain[::-1]


# ---------------------------------------------------------------------------------------------------------------------
# the dataset

class BagDataset(Dataset):
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self._eps = episode_paths(path)
        if not self._eps:
            raise DatasetError(f'{os.path.basename(path) or path} holds no ROS 2 bag (.mcap or .db3).')
        storages = [_storage(e) for e in self._eps]
        self.format = 'rosbag2-sqlite' if storages[0] == 'sqlite3' else 'mcap'
        meta = None if os.path.isfile(self._eps[0]) else _metadata(self._eps[0])
        self.version = f"v{meta['version']}" if meta and meta.get('version') is not None else ''
        self.info = {'bags': len(self._eps), 'storage': sorted(set(storages)), 'metadata': meta or {}}
        names = [os.path.splitext(os.path.basename(os.path.normpath(e)))[0] for e in self._eps]
        dup = len(set(names)) != len(names)
        base = self.path if os.path.isdir(self.path) else os.path.dirname(self.path)
        self._ids = [os.path.relpath(e, base) if dup else n for e, n in zip(self._eps, names)]

    def episodes(self) -> list:
        out = []
        for i, (ep, eid) in enumerate(zip(self._eps, self._ids)):
            length = None
            info = None if os.path.isfile(ep) else _metadata(ep)
            if info:
                for tw in info.get('topics_with_message_count') or []:
                    tm = (tw or {}).get('topic_metadata') or {}
                    if tm.get('name') == '/joint_states':
                        length = int(tw.get('message_count') or 0) or None
            out.append(EpisodeRef(i, eid, length, {'bag': os.path.basename(os.path.normpath(ep)),
                                                   'storage': _storage(ep)}))
        return out

    def truth(self):
        import json
        base = self.path if os.path.isdir(self.path) else os.path.dirname(self.path)
        fp = os.path.join(base, 'dexvantage_truth.json')
        try:
            with open(fp, encoding='utf-8') as f:
                doc = json.load(f)
        except (OSError, ValueError):
            return None
        return doc if isinstance(doc, dict) else None

    def _open(self, ep):
        try:
            return RB.BagReader(ep)
        except FileNotFoundError:
            raise DatasetError(f'The bag {os.path.basename(ep)} is missing.') from None
        except ImportError as e:
            raise DatasetError(f'{e}. Install the bags extra: pip install "dexvantage[bags]".') from None
        except ValueError as e:
            raise DatasetError(f'The bag {os.path.basename(os.path.normpath(ep))} cannot be read: '
                               f'{_plain(e, ep)}.') from None

    def load(self, index: int) -> Episode:
        if not 0 <= index < len(self._eps):
            raise DatasetError(f'The dataset has {len(self._eps)} bags; there is no episode {index + 1}.')
        ep = self._eps[index]
        bag = self._open(ep)
        try:
            return self._episode(index, ep, bag)
        except ImportError as e:
            raise DatasetError(f'{e}. Install the bags extra: pip install "dexvantage[bags]".') from None
        except ValueError as e:
            if isinstance(e, DatasetError):
                raise
            raise DatasetError(f'The bag {os.path.basename(os.path.normpath(ep))} is damaged: {_plain(e, ep)}.') \
                from None
        finally:
            bag.close()

    def _episode(self, index: int, ep: str, bag) -> Episode:
        topics, counts = bag.topics, bag.counts
        joint_topics = [t for t, ty in topics.items() if ty == _JOINTS]
        joint = '/joint_states' if '/joint_states' in joint_topics else \
            max(joint_topics, key=lambda t: (not re.search(r'command|target|cmd|goal', t), counts.get(t, 0)),
                default=None)
        images = {t: ty for t, ty in topics.items() if ty in (_IMAGE, _CIMAGE)}
        infos = [t for t, ty in topics.items() if ty == _INFO]
        statics = [t for t, ty in topics.items() if ty == _TF and t.endswith('tf_static')]
        timerefs = [t for t, ty in topics.items() if ty == _TIMEREF]
        task_topics = [t for t, ty in topics.items() if ty == _STRING and t.rstrip('/').split('/')[-1] in TASK_TOPICS]
        desc = [t for t, ty in topics.items() if ty == _STRING and t.endswith('robot_description')]
        wanted = ([joint] if joint else []) + list(images) + infos + statics + timerefs + task_topics + desc

        jrows, cams, first_info, tf_tree, tref = [], {}, {}, {}, {}
        task, robot_type, warnings = '', '', list(bag.warnings)
        for topic, t_ns, raw in bag.messages(wanted, decode=False):
            if topic in images:
                c = cams.get(topic)
                if c is None:
                    c = cams[topic] = {'stamp': [], 'log': [], 'frame': None, 'w': None, 'h': None, 'enc': None,
                                       'first': None, 'bad': 0}
                try:
                    st, fr, w, h, enc = _peek(raw, images[topic] == _CIMAGE)
                except (ValueError, struct.error):
                    c['bad'] += 1
                    continue
                c['stamp'].append(st)
                c['log'].append(t_ns)
                if c['frame'] is None:
                    c['frame'], c['w'], c['h'], c['enc'] = fr, w, h, enc
                    if images[topic] == _CIMAGE:
                        c['first'] = raw
                continue
            try:
                msg = bag.decode(topic, raw)
            except ValueError:
                warnings.append(f'{topic}: a message could not be decoded and was skipped')
                continue
            if topic == joint:
                st = RB.stamp_ns(msg['header']['stamp'])
                jrows.append((st, t_ns, list(msg.get('name') or []), list(msg.get('position') or [])))
            elif topic in infos:
                first_info.setdefault(topic, msg)
            elif topic in statics:
                for tr in msg.get('transforms') or []:
                    tf_tree[tr['child_frame_id'].lstrip('/')] = (tr['header']['frame_id'].lstrip('/'),
                                                                RB.transform_matrix(tr))
            elif topic in timerefs:
                tref.setdefault(topic, {})[RB.stamp_ns(msg['header']['stamp'])] = RB.stamp_ns(msg['time_ref'])
            elif topic in task_topics and not task:
                task = str(msg.get('data') or '')
            elif topic in desc and not robot_type:
                m = re.search(r'<robot[^>]*\bname\s*=\s*["\']([^"\']+)', str(msg.get('data') or ''))
                robot_type = m.group(1) if m else ''

        # joint states
        names = []
        for _, _, nm, _ in jrows:
            for n in nm:
                if n not in names:
                    names.append(n)
        col = {n: i for i, n in enumerate(names)}
        N = len(jrows)
        q = np.full((N, len(names)), np.nan)
        held = 0
        last = np.full(len(names), np.nan)
        j_stamp = np.array([r[0] for r in jrows], np.int64)
        j_log = np.array([r[1] for r in jrows], np.int64)
        for i, (_, _, nm, pos) in enumerate(jrows):
            row = last.copy()
            seen = set()
            for n, v in zip(nm, pos):
                row[col[n]] = v
                seen.add(n)
            held += sum(1 for n in names if n not in seen and not np.isnan(last[col[n]]))
            q[i] = last = row
        j_use_header = N > 0 and bool(np.all(j_stamp != 0))
        j_times = j_stamp if j_use_header else j_log
        ref_off = int(np.median(j_stamp - j_log)) if j_use_header else 0

        # cameras: clocks
        streams, depth = {}, {}
        for topic, c in cams.items():
            n = len(c['stamp'])
            fmt = (c['enc'] or '').lower()
            is_depth = (images[topic] == _IMAGE and c['enc'] not in _COLOUR) or 'depth' in fmt or \
                (images[topic] == _CIMAGE and 'png' in fmt and '16' in fmt)
            if is_depth:
                depth[topic] = {'count': n, 'encoding': c['enc'], 'width': c['w'], 'height': c['h']}
                continue
            stamp = np.array(c['stamp'], np.int64)
            log = np.array(c['log'], np.int64)
            hw, hw_src, source = None, None, 'header.stamp'
            if n and np.all(stamp != 0):
                off = int(np.median(stamp - log))
                if abs(off - ref_off) > CLOCK_SPLIT_S * 1e9 and joint is not None:
                    hw, hw_src = stamp, 'header.stamp (a device clock: its offset from the log time differs from ' \
                        'the joint states\' by more than CLOCK_SPLIT_S)'
                    times, source = log + ref_off, 'log time carried onto the joint-state clock'
                else:
                    times = stamp
            else:
                times, source = log + ref_off, 'log time (the driver leaves header.stamp at zero)'
            ns = _namespace(topic)
            for tr_topic, mp in tref.items():
                if ns and (tr_topic.startswith(ns + '/') or _namespace(tr_topic) == ns):
                    vals = np.array([mp.get(int(s), -1) for s in stamp], np.int64)
                    if n and np.mean(vals >= 0) > 0.5:
                        hw, hw_src = vals, f'{tr_topic} (sensor_msgs/TimeReference.time_ref)'
                    break
            streams[topic] = (times, log, hw, hw_src, source)

        starts = ([int(np.min(j_times))] if N else []) + [int(np.min(v[0])) for v in streams.values() if len(v[0])]
        t0 = min(starts) if starts else 0

        def rel(a):
            return (np.asarray(a, np.int64) - t0).astype(np.float64) / 1e9

        cameras = {}
        for topic, (times, log, hw, hw_src, source) in streams.items():
            c = cams[topic]
            w, h = c['w'], c['h']
            info_topic = next((t for t in infos if _namespace(t) == _namespace(topic)), None)
            if info_topic is None:
                info_topic = next((t for t in infos if first_info.get(t, {}).get('header', {}).get('frame_id')
                                   == c['frame']), None)
            intr = _lens(first_info[info_topic]) if info_topic in first_info else None
            if (w is None or h is None) and c['first'] is not None:
                try:
                    img = _decoded_rgb(bag, topic, c['first'], c['enc'])
                    h, w = img.shape[:2]
                except (ValueError, ImportError):
                    pass
            if (w is None or h is None) and intr:
                w, h = intr['width'], intr['height']
            sp = _static_pose(tf_tree, (c['frame'] or '').lstrip('/'))
            extr = {'frame': sp[0], 'pose': _pose_dict(sp[1])} if sp else None
            hw_rel, hw0 = None, None
            if hw is not None:
                ok = hw >= 0
                if np.any(ok):
                    first = int(hw[ok][0])
                    hw_rel = np.where(ok, (hw - first).astype(np.float64) / 1e9, np.nan)
                    hw0 = first / 1e9
            meta = {'topic': topic, 'type': images[topic], 'encoding': c['enc'], 'frame_id': c['frame'],
                    'log_times': rel(log), 'timestamp_source': source, 'messages': len(times),
                    'undecodable_headers': c['bad'], 'camera_info': info_topic,
                    'tf_chain': sp[2] if sp else None, 'hw_timestamp_source': hw_src, 'hw_t0_s': hw0,
                    'source': 'rosbag2'}
            if intr is None and info_topic:
                meta['camera_info_note'] = 'the CameraInfo has an all-zero K (uncalibrated)'
            cameras[topic] = CameraStream(key=topic, width=w, height=h, fps=None, timestamps=rel(times),
                                          hw_timestamps=hw_rel, intrinsics=intr, extrinsics=extr, meta=meta,
                                          _decode=_frame_decoder(ep, topic, c['enc'], len(times)))
        if not task:
            cd = ((None if os.path.isfile(ep) else _metadata(ep)) or {}).get('custom_data') or {}
            task = next((str(cd[k]) for k in TASK_TOPICS if isinstance(cd, dict) and cd.get(k)), '')
        meta = {'bag': os.path.basename(os.path.normpath(ep)), 'storage': bag.storage, 'topics': dict(topics),
                'counts': dict(counts), 'joint_topic': joint,
                'other_joint_topics': [t for t in joint_topics if t != joint],
                't_source': 'header.stamp' if j_use_header else 'log time', 'log_times': rel(j_log),
                't0_ns': t0, 'q_held': int(held), 'tf_static_frames': sorted(tf_tree),
                'warnings': warnings, 'decode_errors': dict(bag.decode_errors), 'source': 'rosbag2'}
        if depth:
            meta['depth_streams'] = depth
        if joint is None:
            warnings.append('the bag has no sensor_msgs/JointState topic: the episode has no joint states')
        return Episode(id=self._ids[index], index=index, t=rel(j_times), q=q, joint_names=names, action=None,
                       action_names=[], fps=None, robot_type=robot_type, task=task, cameras=cameras, meta=meta)


def _plain(e: Exception, ep: str) -> str:
    s = str(e).strip().rstrip('.')
    for p in (os.path.abspath(ep), ep):
        s = s.replace(p + os.sep, '').replace(p, os.path.basename(os.path.normpath(ep)))
    return s


def _frame_decoder(ep: str, topic: str, encoding: str, n: int):
    def decode(indices):
        want = None if indices is None else sorted({int(i) for i in indices})
        if want is not None and want and (want[0] < 0 or want[-1] >= n):
            bad = want[0] if want[0] < 0 else want[-1]
            raise DatasetError(f'Camera {topic} has {n} frames; there is no frame {bad}.')
        if want == []:
            return
        wset = None if want is None else set(want)
        last = None if want is None else want[-1]
        try:
            bag = RB.BagReader(ep)
        except (ValueError, OSError) as e:
            raise DatasetError(f'The bag {os.path.basename(os.path.normpath(ep))} cannot be read: '
                               f'{_plain(e, ep)}.') from None
        try:
            k = -1
            for _, _, raw in bag.messages(topic, decode=False):
                try:
                    _peek(raw, bag.topics[topic] == _CIMAGE)
                except (ValueError, struct.error):
                    continue                        # the same messages load() skipped
                k += 1
                if wset is None or k in wset:
                    try:
                        yield k, _decoded_rgb(bag, topic, raw, encoding)
                    except ValueError as e:
                        raise DatasetError(f'Frame {k} of camera {topic} cannot be decoded ({e}).') from None
                if last is not None and k >= last:
                    break
        finally:
            bag.close()
    return decode


def open_bag(path: str) -> BagDataset:
    """Open a ROS 2 bag, a single .mcap / .db3 file, or a folder of bags (one episode per bag)."""
    return BagDataset(path)


register_reader('mcap', _sniff('mcap'))(open_bag)
register_reader('rosbag2-sqlite', _sniff('sqlite3'))(open_bag)
