"""Checks of the project file: a populated project written to a .dexv and read back with Python's zipfile and json
modules alone (not DexVantage's reader) must validate against the published JSON Schema with an independent validator
(the jsonschema package), and DexVantage's own reader must give back the same content hash. And the LeRobot dataset
round trip (formats.lerobot_roundtrip): tiny synthetic episodes written as LeRobot v2.1 and v3.0 and read back, both by
DexVantage's reader and independently (pyarrow.parquet and json on the files at the paths the format defines, the
videos decoded by the ffmpeg command-line tool), against the source arrays."""
from __future__ import annotations

import importlib.util
import json
import os
import zipfile

import numpy as np

from .suite import Outcome, check

SCHEMA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resources',
                           'project.schema.json')


def reference_project():
    """A project with every container filled: two robots, sensors on a link and on a static frame, volumes, objects
    of each material tag, a requirement, a mesh array and a source file."""
    from ..core import project as P
    from ..core.frames import Pose, quat_from_rpy
    p = P.blank_project('Format reference')
    a = P.Robot(name='Left arm', primary=True)
    a.joints = [{'name': 'j1', 'type': 'revolute', 'parent': 'base', 'child': 'l1', 'axis': [0, 0, 1],
                 'lower': -2.9, 'upper': 2.9}]
    a.source = P.RobotSource('urdf', 'left.urdf', 'ab' * 32, 'BSD-3-Clause', 'reference')
    b = P.Robot(name='Right arm', base=Pose([0, -0.6, 0], quat_from_rpy(0, 0, np.pi)))
    p.robots += [a, b]
    v = p.variant
    v.sensors.append(P.Sensor(name='Wrist', kind='depth', mount=P.Mount('l1', a.uid, Pose([0.05, 0, 0.02]))))
    v.sensors.append(P.Sensor(name='Overhead', kind='camera', slot=1,
                              mount=P.Mount('cage', '', Pose([0.4, 0, 1.2], quat_from_rpy(0, np.pi / 2, 0)))))
    p.workspace.objects += [P.TaskObject(name=f'{m} cube', materials=[m]) for m in P.MATERIAL_TAGS]
    p.workspace.volumes.append(P.Volume(name='Handover', kind='handover', refine=True))
    p.requirements.append(P.Requirement(name='Contact visibility', metric='grasp_visibility.score', op='>=',
                                        threshold=0.95, study='grasp_visibility'))
    p.meshes['l1/visual'] = np.arange(36, dtype=np.float32).reshape(-1, 3)
    p.sources['left.urdf'] = b'<robot name="left"/>'
    return p


def _written(ctx):
    if 'formats.file' in ctx.cache:
        return ctx.cache['formats.file']
    p = reference_project()
    path = os.path.join(ctx.tmp(), 'reference.dexv')
    p.save(path)
    ctx.cache['formats.file'] = (p, path)
    return p, path


@check('formats.schema', 'formats', 'Project file against its JSON Schema',
       'number of schema violations in project.json of a fully populated project, read with zipfile and json only',
       'the published JSON Schema (resources/project.schema.json, also docs/project.schema.json) through the '
       'jsonschema package\'s Draft 2020-12 validator', 0, 'violations', refs=('json_schema', 'zip_appnote'))
def _schema(ctx):
    import jsonschema
    _, path = _written(ctx)
    with zipfile.ZipFile(path) as z:
        doc = json.loads(z.read('project.json').decode('utf-8'))
        names = z.namelist()
    with open(SCHEMA_FILE, encoding='utf-8') as fh:
        schema = json.load(fh)
    jsonschema.Draft202012Validator.check_schema(schema)
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(doc))
    return Outcome(len(errors), {'members': sorted(names), 'first_errors': [e.message for e in errors[:5]]})


@check('formats.round_trip', 'formats', 'Project file round trip',
       'number of content sections whose hash differs after save and load, plus array or source files not read back '
       'bit for bit', 'the project in memory before saving (core.provenance content hash per section)', 0,
       'differences', refs=('zip_appnote',))
def _round_trip(ctx):
    from ..core import provenance as PV
    from ..core.project import Project
    p, path = _written(ctx)
    q = Project.load(path)
    a, b = PV.project_digest(p), PV.project_digest(q)
    diff = [k for k in a['sections'] if a['sections'][k] != b['sections'].get(k)]
    arrays = [k for k in p.meshes if k not in q.meshes or q.meshes[k].dtype != p.meshes[k].dtype
              or not np.array_equal(q.meshes[k], p.meshes[k])]
    files = [k for k in p.sources if q.sources.get(k) != p.sources[k]]
    return Outcome(len(diff) + len(arrays) + len(files), {'sections_differ': diff, 'arrays_differ': arrays,
                                                          'sources_differ': files, 'sha256': b['sha256']})



# ---------------------------------------------------------------------------------------------------------------
# LeRobot v2.1 and v3.0 (core/datasets/lerobot.py)

LEROBOT_CAM = 'observation.images.wrist'
LEROBOT_PSNR_DB = 30.0      # per-frame floor for the lossy codec (see the check's note for why)
LEROBOT_TS_TOL_S = 0.5 / 90000 + 1e-9   # half a tick of the 90 kHz time base of irregular cameras


def lerobot_pattern(i: int, h: int = 48, w: int = 64) -> np.ndarray:
    """A smooth test image whose bright blob and colour ramp move with the frame number i (so a frame read back in
    the wrong place is told apart from its neighbours)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    cx = 8 + (i * 11) % (w - 16)
    cy = 10 + (i * 7) % (h - 20)
    blob = 140 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 4.0 ** 2))
    r = 30 + 150 * xx / w + blob
    g = 50 + 120 * yy / h + 0.5 * blob + 30 * np.sin(i * 0.9)
    b = 90 + 60 * np.sin(xx / w * 3 + i * 0.7) + 0.3 * blob
    return np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8)


def lerobot_reference_episodes(rng, fps: int = 10) -> list:
    """Three tiny episodes: 12 and 9 rows with regular camera frames, and 8 rows whose camera drops a frame and runs
    one frame 13 ms late (irregular timestamps); six joints, actions, an extra two-column feature, two tasks."""
    from ..core.datasets import episode as E
    out = []
    for j, (n, cam_t) in enumerate([(12, None), (9, None),
                                    (8, np.array([0.0, 0.1, 0.2, 0.4, 0.5, 0.613, 0.7]))]):
        t = np.arange(n) / fps
        ts = t.copy() if cam_t is None else cam_t
        q = rng.uniform(-2.0, 2.0, size=(n, 6))
        frames = [lerobot_pattern(37 * j + i) for i in range(len(ts))]

        def dec(idx, frames=frames):
            for i in (range(len(frames)) if idx is None else idx):
                yield i, frames[i]
        cam = E.CameraStream(LEROBOT_CAM, 64, 48, float(fps), ts, _decode=dec)
        cam.meta['source_frames'] = frames
        out.append(E.Episode(f'ref{j}', j, t, q, [f'joint_{k}' for k in range(6)], q + rng.normal(0, 0.01, q.shape),
                             [f'joint_{k}.target' for k in range(6)], float(fps), 'reference_arm',
                             ('pick the cube', 'place the cube')[j % 2], {LEROBOT_CAM: cam},
                             {'columns': {'observation.effort': rng.normal(size=(n, 2)).astype(np.float32)}}))
    return out


LEROBOT_EXTRA = {'observation.effort': {'dtype': 'float32', 'shape': [2], 'names': ['gripper', 'wrist']}}


def _psnr(a, b) -> float:
    mse = float(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2))
    return float('inf') if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def _ffmpeg_frames(path: str, h: int, w: int):
    """(pts seconds, frames) of a video decoded by the ffmpeg command-line tool, or None when it is not on PATH."""
    import shutil
    import subprocess
    if not (shutil.which('ffmpeg') and shutil.which('ffprobe')):
        return None
    pts = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'frame=pts_time',
                          '-of', 'csv=p=0', path], capture_output=True, text=True, check=True, timeout=60).stdout
    t = np.array([float(x.strip().rstrip(',')) for x in pts.split() if x.strip()])
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-fps_mode', 'passthrough', '-f', 'rawvideo',
                          '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True, timeout=60).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    return t, fr


def _lerobot_independent(root: str, version: str, src: list) -> tuple[list, dict]:
    """Mismatches found reading the written dataset without DexVantage's reader: json and pyarrow.parquet on the files
    at the format's paths, ffmpeg for the videos."""
    import pyarrow.parquet as pq
    from ..core.datasets import lerobot as L
    bad, det = [], {'psnr_ffmpeg_min_db': None, 'ffmpeg': True}
    info = json.load(open(os.path.join(root, L.INFO_PATH), encoding='utf-8'))
    if info.get('codebase_version') != version:
        bad.append(f'info.json codebase_version {info.get("codebase_version")}')
    feats = info.get('features', {})
    want = {'observation.state': ('float32', [6]), 'action': ('float32', [6]), 'observation.effort': ('float32', [2]),
            'timestamp': ('float32', [1]), 'frame_index': ('int64', [1]), 'episode_index': ('int64', [1]),
            'index': ('int64', [1]), 'task_index': ('int64', [1]), LEROBOT_CAM: ('video', [48, 64, 3])}
    for k, (dt, sh) in want.items():
        if k not in feats or feats[k].get('dtype') != dt or list(feats[k].get('shape') or []) != sh:
            bad.append(f'info.json feature {k}')
    if feats.get('observation.state', {}).get('names') != src[0].joint_names:
        bad.append('info.json observation.state names')
    if info.get('total_frames') != sum(len(e.t) for e in src) or info.get('total_episodes') != len(src):
        bad.append('info.json totals')
    if version == 'v2.1':
        rows = [json.loads(x) for x in open(os.path.join(root, L.V21_EPISODES), encoding='utf-8') if x.strip()]
        tasks = {d['task_index']: d['task'] for d in
                 (json.loads(x) for x in open(os.path.join(root, L.V21_TASKS), encoding='utf-8') if x.strip())}
    else:
        rows = pq.read_table(os.path.join(root, L.V30_EPISODES_PATH.format(chunk_index=0, file_index=0))).to_pylist()
        tt = pq.read_table(os.path.join(root, L.V30_TASKS)).to_pylist()
        tasks = {d['task_index']: d['task'] for d in tt}
    psnr_min = float('inf')
    for e, row in zip(src, rows):
        j, n = e.index, len(e.t)
        if row['length'] != n or row['tasks'] != [e.task]:
            bad.append(f'episode {j} index row')
        if version == 'v2.1':
            dp = os.path.join(root, L.V21_DATA_PATH.format(episode_chunk=0, episode_index=j))
            tab = pq.read_table(dp)
            vp = os.path.join(root, L.V21_VIDEO_PATH.format(episode_chunk=0, video_key=LEROBOT_CAM, episode_index=j))
            lo, hi = 0.0, float('inf')
        else:
            dp = os.path.join(root, L.V30_DATA_PATH.format(chunk_index=row['data/chunk_index'],
                                                          file_index=row['data/file_index']))
            tab = pq.read_table(dp).slice(row['dataset_from_index'], n)
            vp = os.path.join(root, L.V30_VIDEO_PATH.format(video_key=LEROBOT_CAM,
                                                           chunk_index=row[f'videos/{LEROBOT_CAM}/chunk_index'],
                                                           file_index=row[f'videos/{LEROBOT_CAM}/file_index']))
            lo, hi = row[f'videos/{LEROBOT_CAM}/from_timestamp'], row[f'videos/{LEROBOT_CAM}/to_timestamp']
        d = tab.to_pydict()
        pairs = {'observation.state': e.q, 'action': e.action, 'timestamp': e.t,
                 'observation.effort': e.meta['columns']['observation.effort']}
        for k, ref in pairs.items():
            if not np.array_equal(np.asarray(d[k], np.float32), np.asarray(ref).astype(np.float32)):
                bad.append(f'episode {j} column {k} (pyarrow)')
        if d['frame_index'] != list(range(n)) or set(d['episode_index']) != {j} or \
                tasks.get(d['task_index'][0]) != e.task:
            bad.append(f'episode {j} index columns (pyarrow)')
        got = _ffmpeg_frames(vp, 48, 64)
        if got is None:
            det['ffmpeg'] = False
            continue
        t_all, fr_all = got
        sel = (t_all >= lo - 1e-4) & (t_all < hi - 1e-4)
        cam = e.cameras[LEROBOT_CAM]
        src_frames = cam.meta['source_frames']
        if int(sel.sum()) != len(src_frames):
            bad.append(f'episode {j} frame count (ffmpeg {int(sel.sum())}, source {len(src_frames)})')
            continue
        if np.max(np.abs(t_all[sel] - lo - cam.timestamps)) > LEROBOT_TS_TOL_S + 1e-6:   # ffprobe prints 6 decimals
            bad.append(f'episode {j} frame times (ffprobe)')
        for a, b in zip(fr_all[sel], src_frames):
            psnr_min = min(psnr_min, _psnr(a, b))
    det['psnr_ffmpeg_min_db'] = None if psnr_min == float('inf') else round(float(psnr_min), 2)
    if det['ffmpeg'] and psnr_min < LEROBOT_PSNR_DB:
        bad.append(f'ffmpeg-decoded frames below {LEROBOT_PSNR_DB} dB')
    return bad, det


def _lerobot_reader(root: str, src: list) -> tuple[list, dict]:
    """Mismatches between DexVantage's reader and the source arrays."""
    from ..core.datasets import open_dataset
    bad = []
    psnr_min, order_margin, ts_err, q_err = float('inf'), float('inf'), 0.0, 0.0
    with open_dataset(root) as ds:
        refs = ds.episodes()
        if [r.length for r in refs] != [len(e.t) for e in src]:
            bad.append('episode lengths')
        for e, r in zip(src, refs):
            g = ds.load(r.index)
            j = e.index
            for name, a, b in (('q', g.q, e.q), ('action', g.action, e.action), ('t', g.t, e.t),
                               ('observation.effort', g.meta['columns'].get('observation.effort'),
                                e.meta['columns']['observation.effort'])):
                if a is None or not np.array_equal(np.asarray(a), np.asarray(b).astype(np.float32).astype(a.dtype)):
                    bad.append(f'episode {j} {name}')
            q_err = max(q_err, float(np.max(np.abs(g.q - e.q))))
            if g.joint_names != e.joint_names or g.action_names != e.action_names or g.task != e.task or \
                    g.robot_type != e.robot_type or g.fps != e.fps:
                bad.append(f'episode {j} names, task, robot type or fps')
            cam, c0 = g.cameras.get(LEROBOT_CAM), e.cameras[LEROBOT_CAM]
            if cam is None or cam.n_frames != c0.n_frames:
                bad.append(f'episode {j} camera frame count')
                continue
            ts_err = max(ts_err, float(np.max(np.abs(cam.timestamps - c0.timestamps))))
            frames = dict(cam.frames())
            sf = c0.meta['source_frames']
            for i in range(len(sf)):
                p = _psnr(frames[i], sf[i])
                psnr_min = min(psnr_min, p)
                for k in (i - 1, i + 1):
                    if 0 <= k < len(sf):
                        order_margin = min(order_margin, p - _psnr(frames[i], sf[k]))
    if ts_err > LEROBOT_TS_TOL_S:
        bad.append(f'camera timestamps off by {ts_err:.2e} s')
    if psnr_min < LEROBOT_PSNR_DB:
        bad.append(f'frames below {LEROBOT_PSNR_DB} dB')
    if order_margin <= 0:
        bad.append('a frame read back closer to its neighbour than to itself')
    return bad, {'psnr_reader_min_db': round(float(psnr_min), 2), 'order_margin_db': round(float(order_margin), 2),
                 'camera_time_error_s': ts_err, 'state_error_vs_float64_source': q_err}


@check('formats.lerobot_roundtrip', 'formats', 'LeRobot v2.1 and v3.0 round trip',
       'number of mismatches when three synthetic episodes (state, action, an extra feature, a camera with regular and '
       'with dropped and late frames) are written as LeRobot v2.1 and v3.0 and read back: columns not bit-exact after '
       'the float32 cast, camera times off by more than half a 90 kHz tick, frame counts, frames below 30 dB PSNR or '
       'closer to a neighbouring source frame, info.json features and totals',
       'the source arrays; the files read independently of DexVantage\'s reader with json and pyarrow.parquet at the '
       'paths the LeRobot source defines, and the videos decoded by the ffmpeg and ffprobe command-line tools',
       0, 'mismatches', refs=('lerobot_dataset_format', 'apache_parquet', 'ffmpeg_cli'))
def _lerobot_roundtrip(ctx):
    from ..core.datasets import lerobot as L
    src = lerobot_reference_episodes(ctx.rng('formats.lerobot_roundtrip'))
    detail, bad = {}, []
    for version in L.SUPPORTED:
        root = os.path.join(ctx.tmp(), f'lerobot_{version}')
        summary = L.write_lerobot(root, src, version, 10, 'reference_arm', LEROBOT_EXTRA)
        b1, d1 = _lerobot_reader(root, src)
        b2, d2 = _lerobot_independent(root, version, src)
        bad += [f'{version}: {x}' for x in b1 + b2]
        detail[version] = {'codec': summary['codec'], 'files': summary['files'], **d1, **d2}
    try:
        import lerobot  # noqa: F401
        have = True
    except ImportError:
        have = False
    note = ('The lerobot package is not installed, so this check uses DexVantage\'s own reader and an independent '
            'read with pyarrow and ffmpeg, against the format as read from the LeRobot source at commit d490e47 '
            '(v3.0) and tag v0.3.3 (v2.1); LeRobot\'s own loader is not run. ' if not have else '')
    note += ('Tolerances: columns are compared bit for bit after the float32 cast the format stores; camera times '
             'within half a tick of the 1/90000 s time base used for irregular cameras (regular ones are exact); '
             'frames at 30 dB PSNR or better, because the videos are lossy (LeRobot\'s defaults: AV1 or H.264, '
             'yuv420p, crf 30) and 30 dB, an RMS error of 8 grey levels, is below what was measured on this pattern '
             '(worst frame 33.2 dB at v3.0 and 35.7 dB at v2.1 with libsvtav1, 2026-11-07) with margin for other '
             'encoder builds.')
    detail['mismatches'] = bad
    return Outcome(len(bad), detail, note=note)


# ---------------------------------------------------------------------------------------------------------------------
# ROS 2 bags and RLDS TFRecords (the work is in formats_bags.py)

@check('formats.mcap_roundtrip', 'formats', 'ROS 2 bag (MCAP) round trip',
       'messages whose topic, log time or bytes differ, chunk / data / summary CRC failures, and joint states, stamps '
       'or camera poses the dataset reader does not give back exactly, over every installed chunk compression',
       'an MCAP walk written again from the specification (zlib CRC-32, zstandard / lz4 directly), the mcap package '
       'with CRC validation when installed, and a JointState laid out by hand in XCDR1', 0, 'mismatches',
       refs=('mcap_spec', 'mcap_python', 'rosbag2_mcap', 'omg_xtypes_cdr'))
def _mcap_roundtrip(ctx):
    from . import formats_bags
    err, detail = formats_bags.mcap_roundtrip(ctx)
    return Outcome(err, detail, note='Zero: a container format either carries every byte or it is wrong; the reader '
                                     'must return the written joint states and stamps bit for bit.')


@check('formats.tfrecord_crc', 'formats', 'TFRecord CRC-32C',
       'CRC-32C values that differ from the published vectors or from a bit-at-a-time reference, masked CRCs that '
       'differ from TensorFlow\'s formula, and TFRecord framing read wrongly or damage not refused',
       'RFC 3720 appendix B.4 test vectors and the CRC catalogue check value of CRC-32/ISCSI; a bitwise Castagnoli '
       'CRC and the TFRecord framing written again in the check', 0, 'mismatches',
       refs=('rfc3720', 'crc_catalogue', 'tfrecord_format'))
def _tfrecord_crc(ctx):
    from . import formats_bags
    err, detail = formats_bags.tfrecord_crc(ctx)
    return Outcome(err, detail, note='Zero: a checksum is exact; one wrong bit would accept damaged records or refuse '
                                     'good ones.')


# ---------------------------------------------------------------------------------------------------------------------
# the MuJoCo scene written to disk (core/sim/scene.py save_scene, Simulate ▸ Build the MuJoCo scene…)

MJCF_SAMPLES = ('ur5e', 'so100', 'dexhand')


def mjcf_scene_errors(project, folder: str, rng) -> tuple[float, float, dict]:
    """(largest optical-frame position error (m), largest rotation error (rad), detail) of the scene saved to folder
    and loaded by MuJoCo's own XML parser from disk, posed at a random joint state set by joint name, against
    DexVantage's own kinematics (camrender.optical_pose: RobotModel FK of the stored robot files, the mount and the
    imager's pose in the sensor body); and the intrinsics read back from the loaded model against the lens models."""
    import mujoco

    from ..core import camrender as CR
    from ..core import library as LB
    from ..core.robot.model import get_model
    from ..core.sim.scene import build_scene, save_scene
    from ..core.studies import coverage as CV
    from .mjrender import camera_model
    path = save_scene(project, folder)
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    b = build_scene(project)                         # only for the robots' name prefixes and the camera names
    q = {}
    for r in project.robots:
        if r.uid not in b.robots:
            continue
        rm = get_model(project, r.uid)
        lo, hi = np.asarray(rm.lower, float), np.asarray(rm.upper, float)
        home = np.asarray(rm.home, float)
        lo = np.where(np.isfinite(lo), lo, home - np.pi)
        hi = np.where(np.isfinite(hi), hi, home + np.pi)
        qr = lo + (hi - lo) * rng.uniform(0.2, 0.8, len(home))
        q[r.uid] = qr
        for name, v in zip(rm.joint_names, qr):
            j = m.joint(b.robots[r.uid]['prefix'] + name)
            d.qpos[j.qposadr[0]] = v
    mujoco.mj_forward(m, d)
    flip = np.diag([1.0, -1.0, -1.0])
    sensors = {s.uid: s for s in project.variant.sensors}
    pos_err, rot_err, k_err, rows = 0.0, 0.0, 0.0, []
    for (uid, iname), c in b.cameras.items():
        s = sensors[uid]
        im = next(i for i in LB.sensor_imagers(s, CV.sensor_mode(s, {})) if i.name == iname)
        T = CR.optical_pose(project, s, im, q).matrix()
        cid = m.camera(c.name).id
        R_mj = d.cam_xmat[cid].reshape(3, 3) @ flip
        e_p = float(np.linalg.norm(d.cam_xpos[cid] - T[:3, 3]))
        cos = (np.trace(R_mj.T @ T[:3, :3]) - 1) / 2
        e_r = float(np.arccos(np.clip(cos, -1.0, 1.0)))
        cm = camera_model(m, c.name, c.model.width, c.model.height)
        e_k = max(abs(cm.fx - c.model.fx), abs(cm.fy - c.model.fy), abs(cm.cx - c.model.cx), abs(cm.cy - c.model.cy))
        pos_err, rot_err, k_err = max(pos_err, e_p), max(rot_err, e_r), max(k_err, e_k)
        rows.append({'camera': c.name, 'position_m': e_p, 'rotation_rad': e_r, 'intrinsics_px': e_k})
    return pos_err, rot_err, {'bodies': int(m.nbody), 'geoms': int(m.ngeom), 'meshes': int(m.nmesh),
                              'cameras': rows, 'intrinsics_px_max': k_err, 'files': len(os.listdir(
                                  os.path.join(folder, 'assets')))}


@check('formats.mjcf_scene', 'formats', 'MuJoCo scene file loads in MuJoCo and poses like the project',
       'largest distance (m) between each scene camera\'s optical centre in the saved MJCF, loaded from disk by '
       'mujoco.MjModel.from_xml_path and posed at a random joint state set by joint name, and the same camera posed '
       'by DexVantage\'s kinematics; also every camera\'s rotation within 5 × 10⁻⁶ rad and intrinsics within 10⁻² px; '
       'the UR5e, SO-100 and dexhand samples',
       'core/camrender.optical_pose (RobotModel forward kinematics of the stored robot files, the mount and the '
       'imager pose) and the imagers\' lens models; MuJoCo\'s XML parser reading the written files', 2e-6, 'm',
       refs=('mujoco_camera_317', 'mujoco_xml'), cost='study')
def _mjcf_scene(ctx):
    from ..core.robot.samples import sample_project
    rng = ctx.rng('formats.mjcf_scene')
    worst_p, worst_r, detail = 0.0, 0.0, {}
    k_worst = 0.0
    for name in MJCF_SAMPLES:
        p = sample_project(name)
        e_p, e_r, det = mjcf_scene_errors(p, os.path.join(ctx.tmp(), f'scene_{name}'), rng)
        worst_p, worst_r = max(worst_p, e_p), max(worst_r, e_r)
        k_worst = max(k_worst, det['intrinsics_px_max'])
        detail[name] = det
    detail['rotation_rad_max'] = worst_r
    detail['intrinsics_px_max'] = k_worst
    return Outcome(worst_p, detail, ok=worst_r <= 5e-6 and k_worst <= 1e-2,
                   note='The file is the one Simulate ▸ Build the MuJoCo scene… writes, read back by MuJoCo from '
                        'disk with its assets folder, so a missing mesh, a wrong path or a pose lost in writing '
                        'shows here. MuJoCo 3.17.0\'s XML writer (MjSpec.to_xml, no precision option) writes 6 '
                        'significant digits: half a unit of the 6th digit is 5 × 10⁻⁷ m on a coordinate under 1 m, '
                        'about 10⁻⁶ rad on a quaternion and 0.005 px on a focal length of 1 000 to 9 999 px '
                        '(focalpixel "1396.81"). The tolerances, 2 µm, 5 × 10⁻⁶ rad and 0.01 px, allow that '
                        'rounding along a few links of a chain (measured 2026-11-07: 5.8 × 10⁻⁷ m, 1.1 × 10⁻⁶ rad, '
                        '1.9 × 10⁻³ px); a pose wrong by a link offset is millimetres. In memory, before writing, '
                        'the scene matches within 10⁻⁴ px (groundtruth.scene_intrinsics).')


# ---------------------------------------------------------------------------------------------------------------------
# exporters part A (core/exporters: mjcf, ros2, calibration, masks; the work is in formats_exports.py)

@check('formats.mjcf_snippets', 'formats', 'MJCF camera snippets load in MuJoCo and pose like the project',
       'largest distance (m) between each camera of the exported <camera>/<site> snippets, pasted into its body of '
       'the robot\'s own MJCF and loaded by MuJoCo, posed at a random joint state by joint name, and the same camera '
       'posed by DexVantage; also every rotation within 10⁻⁹ rad and the intrinsics MuJoCo compiles within 10⁻³ px; '
       'the UR5e, SO-100 and dexhand samples',
       'core/camrender.optical_pose (RobotModel forward kinematics, the mount and the imager pose) and the imagers\' '
       'lens models; MuJoCo\'s XML parser reading the snippets', 1e-9, 'm',
       refs=('mujoco_camera_317', 'mujoco_xml'), cost='study')
def _mjcf_snippets(ctx):
    from . import formats_exports as FE
    err, ok, detail = FE.mjcf_snippets(ctx)
    return Outcome(err, detail, ok=ok,
                   note='The snippets are written with 17 significant digits (they read back to the same float64), '
                        'so the poses agree to rounding (10⁻⁹ m and rad leave room for a chain of a dozen bodies); '
                        'MuJoCo stores the compiled intrinsics in float32 (about 6 × 10⁻⁵ px at a 1 000 px focal '
                        'length), hence 10⁻³ px. The pasted file is also loaded through MuJoCo\'s own writer to show '
                        'every camera is found under its body; the figures come from each block parsed by MuJoCo and '
                        'attached to the robot\'s model in memory, since that writer keeps 6 digits. The full scene is '
                        'formats.mjcf_scene.')


@check('formats.urdf_roundtrip', 'formats', 'URDF parses and its sensor frames match DexVantage\'s kinematics',
       'largest distance (m) between every sensor body frame and optical frame of the exported rig.urdf, posed at a '
       'random joint state (1) by a forward-kinematics walk of the XML with urdfdom\'s origin and RPY rules and (2) '
       'by MuJoCo\'s URDF compiler, and the same frames from DexVantage; also every rotation within 10⁻⁹ rad and no '
       'structural problem (one root, every link reached once, complete <limit> elements); the three samples',
       'scenegeom.sensor_world_pose and camrender.optical_pose; Python\'s ElementTree with urdfdom\'s setFromRPY '
       'quaternion and Rodrigues\' rotation written again; MuJoCo\'s URDF compiler', 1e-9, 'm',
       refs=('urdfdom', 'urdf', 'mujoco_urdf', 'python_ast'), cost='study')
def _urdf_roundtrip(ctx):
    from . import formats_exports as FE
    err, ok, detail = FE.urdf_roundtrip(ctx)
    return Outcome(err, detail, ok=ok,
                   note='The URDF is written from the MuJoCo model DexVantage compiles, joint by joint (anchors and '
                        'qpos0 folded into the origins), with 17 significant digits; two independent posers, an XML '
                        'walk with urdfdom\'s conventions and MuJoCo\'s URDF compiler, must both agree with '
                        'DexVantage to rounding (10⁻⁹ m over chains of up to 30 joints). A wrong joint anchor, a sign '
                        'of qpos0 or an RPY order would be millimetres or degrees.')


@check('formats.ros2_tf', 'formats', 'Static transform launch file and tf2 tree pose the cameras like DexVantage',
       'largest distance (m) between each optical frame posed from the exported static_transforms.launch.py (its '
       'static_transform_publisher arguments read with Python\'s ast) on its parent link at a random joint state, or '
       'from tf_tree.yaml\'s world pose at the export\'s joint state (PyYAML), and the frame posed by DexVantage; '
       'rotations within 10⁻⁹ rad; the launch file and the tree list the same frames; the three samples',
       'camrender.optical_pose; RobotModel.link_pose for the parent link; tf2_ros\'s argument names', 1e-9, 'm',
       refs=('tf2_static_publisher', 'pyyaml', 'python_ast', 'rep103'), cost='study')
def _ros2_tf(ctx):
    from . import formats_exports as FE
    err, ok, detail = FE.ros2_tf(ctx)
    return Outcome(err, detail, ok=ok,
                   note='The transforms are numbers written with 17 significant digits: any error beyond rounding is '
                        'a wrong frame, parent or quaternion order (tf2 takes x, y, z, w).')


@check('formats.opencv_yaml', 'formats', 'OpenCV YAML read back by cv2.FileStorage equals the source',
       'largest absolute difference between what cv2.FileStorage reads back from each imager\'s YAML (image size, K, '
       'the 4 × 4 extrinsics) and cameras.json and the source; and every lens projected through cv2.projectPoints or '
       'cv2.fisheye.projectPoints with the written K and D within 10⁻⁶ px of DexVantage\'s lens model when the '
       'mapping is exact, within the stated error when it is an approximation; seven lens models on the UR5e '
       'sample\'s six imagers',
       'the source CameraModel and transforms; OpenCV\'s own projection functions', 0.0, 'abs. difference',
       refs=('opencv_filestorage', 'opencv_calib3d', 'opencv_fisheye'), cost='study')
def _opencv_yaml(ctx):
    from . import formats_exports as FE
    err, ok, detail = FE.opencv_yaml(ctx)
    return Outcome(err, detail, ok=ok,
                   note='Zero: cv2.FileStorage writes doubles with 17 significant digits, so a matrix reads back bit '
                        'for bit (measured 0 on 2026-11-07). The projection bound of 10⁻⁶ px is far above float64 '
                        'rounding (10⁻¹² px measured) and far below any wrong coefficient order; the equisolid and '
                        'stereographic lenses have no OpenCV form and must stay within the error the file states '
                        '(measured 5 × 10⁻⁹ and 4 × 10⁻⁴ px on these lenses).')


@check('formats.kalibr_camchain', 'formats', 'Kalibr camchain parsed back equals the rig',
       'largest absolute difference between camchain.yaml read with PyYAML (intrinsics [fu fv pu pv], exact '
       'distortion terms, T_cn_cnm1) and the chain from DexVantage\'s optical poses, inv(T_world_cn) · '
       'T_world_cnm1; also Kalibr\'s keys, model names and coefficient counts (ConfigReader); seven lens models on '
       'the UR5e sample and the sample itself',
       'camrender.optical_pose and the lens models; Kalibr\'s ConfigReader rules', 0.0, 'abs. difference',
       refs=('kalibr_yaml', 'pyyaml'), cost='study')
def _kalibr_camchain(ctx):
    from . import formats_exports as FE
    err, ok, detail = FE.kalibr_camchain(ctx)
    return Outcome(err, detail, ok=ok,
                   note='Zero: the numbers are written in Python\'s shortest round-trip form, which YAML reads back '
                        'to the same float64.')


@check('formats.masks_vs_occlusion', 'formats', 'Exported masks match the occlusion study per pixel',
       'share of sampled pixels (every 4th row and column; 8th in a quick run) where the exported self-occlusion PNG '
       'differs from exact ray casting against the occlusion study\'s occluders (robot hit nearer than any other); '
       'also within 0.5 % of the occlusion study\'s cube map (n = 2048) and the depth-validity codes within 1 % of '
       'the conditions evaluated again per pixel by exact ray casting; the UR5e sample\'s two cameras and two depth '
       'sensors',
       'core/raycast.py (Möller-Trumbore on a uniform grid) on scenegeom.sensor_occluders; core/occlusion.CubeMap; '
       'the depth-validity conditions written again', 1e-3, 'share of pixels',
       refs=('moller1997', 'amanatides_woo1987', 'greene1986'), cost='study')
def _masks_vs_occlusion(ctx):
    from . import formats_exports as FE
    err, ok, detail = FE.masks_vs_occlusion(ctx)
    return Outcome(err, detail, ok=ok,
                   note='The masks come from the Camera view\'s renderer, which tests each pixel-centre ray against '
                        'every triangle exactly, so exact ray casting should agree but for rays through a shared '
                        'edge (10⁻³; measured 0 of 187 200 pixels on 2026-11-07). The cube map answers per texel '
                        '(0.03° at 2048), so silhouettes may differ by a texel: 0.5 % (measured 0.07 %). The '
                        'depth codes use the study\'s cube maps for the other imager\'s line of sight: 1 % (measured '
                        '0.25 %, at silhouettes; at the study\'s own 512 texels it was 4.4 %, the reason the masks use '
                        '2048).')


if importlib.util.find_spec('gmsh') is not None:
    @check('formats.bracket_step', 'formats', 'Bracket STEP read back by OpenCASCADE equals its mesh',
           'relative difference between the volume of the exported bracket STEP, read back by gmsh\'s OpenCASCADE '
           'kernel, and the volume of the faceted mesh it was written from (divergence theorem); also exactly one '
           'volume and the bounding box within 10⁻³ mm; the UR5e sample\'s wrist bracket at a 2 mm grid',
           'the triangle mesh (core/meshing.py) by the divergence theorem; OpenCASCADE\'s own STEP reader and mass '
           'properties', 1e-6, 'relative', refs=('step_ap', 'gmsh'), cost='study')
    def _bracket_step(ctx):
        from . import formats_exports as FE
        err, ok, detail = FE.bracket_step(ctx)
        return Outcome(err, detail, ok=ok,
                       note='The STEP is the mesh itself (each triangle a planar face, sewn into one shell), so the '
                            'volumes agree to the STEP file\'s number format and OpenCASCADE\'s sewing tolerance; '
                            '10⁻⁶ of the volume is far below a missing or flipped face (a face of a 2 mm grid is '
                            'about 10⁻⁴ of a bracket). Registered only when gmsh is installed.')


# ---------------------------------------------------------------------------------------------------------------------
# Isaac Lab and Isaac Sim, LeRobot train configs, reports (core/exporters/isaac.py, lerobot.py, report.py; the work is
# in formats_isaac_lerobot.py)

EXPORT_B_SAMPLES = ('ur5e', 'so100', 'dexhand')


@check('formats.isaac_usd', 'formats', 'Isaac USD camera prims re-opened match the cameras',
       'largest pixel gap between points projected through each exported camera prim (pxr\'s Gf.Camera frustum of '
       'the prim as re-opened, files written by pxr and by the text writer) and through the imager\'s pinhole part; '
       'also every prim\'s world transform within 10⁻⁹ m and 10⁻⁶ rad, focal length and apertures within 10⁻⁶ '
       'relative, and its OpenCV lens attributes; the three samples, two of their cameras given a Brown-Conrady and '
       'a Kannala-Brandt unit calibration',
       'core/camrender.optical_pose and the imagers\' lens models (optics.CameraModel); OpenUSD\'s own camera '
       'model (pxr UsdGeom.Camera, Gf.Camera)', 1e-3, 'px', refs=('openusd_camera', 'isaaclab_camera_v300'),
       cost='study')
def _isaac_usd(ctx):
    from ..core.exporters import isaac as I
    from ..core.robot.samples import sample_project
    from . import formats_isaac_lerobot as V
    have = I.have_pxr()
    worst = {'proj_px': 0.0, 'pose_m': 0.0, 'pose_rad': 0.0, 'lens_rel': 0.0}
    detail, bad = {}, []
    for name in EXPORT_B_SAMPLES:
        p = V.lens_variety(sample_project(name))
        d = V.isaac_usd_errors(p, os.path.join(ctx.tmp(), f'isaac_usd_{name}')) if have else \
            V.isaac_usd_text_errors(p, '')
        for k in worst:
            worst[k] = max(worst[k], d[k])
        bad += [f'{name}: {m}' for m in d['mismatches']]
        detail[name] = d
    detail.update(worst=worst, mismatches=bad, parser='pxr' if have else 'text')
    ok = worst['pose_m'] <= 1e-9 and worst['pose_rad'] <= 1e-6 and worst['lens_rel'] <= 1e-6 and not bad
    note = ('USD stores focalLength, the apertures and the OpenCV attributes as 32-bit floats (a relative error up to '
            '6 × 10⁻⁸, so about 10⁻⁴ px at a 1 400 px focal length across a 1 920 px image); 10⁻³ px allows that '
            'with margin and is far below a sign or convention error (an aperture offset of the wrong sign moves the '
            'principal point by twice its offset, tens of pixels on the calibrated cameras here). Measured '
            '2026-11-07: 2.6 × 10⁻⁵ px. Transforms are matrix4d (64-bit): 10⁻⁹ m and 10⁻⁶ rad are far below a '
            'frame-convention error (a quarter or half turn), measured 5.6 × 10⁻¹⁷ m and 5.5 × 10⁻¹⁶ rad.')
    if not have:
        note = ('usd-core is not installed: the text-written file is read by the check\'s own reader and the pinhole '
                'its attributes define is computed by GfCamera\'s frustum formula written out in the check; install '
                'the usd extra for the pxr round trip. ') + note
    return Outcome(worst['proj_px'], detail, ok=ok, note=note)


@check('formats.isaac_config', 'formats', 'Isaac Lab camera configs compile and their values match',
       'largest position difference (m) between each camera\'s CameraCfg offset in the generated Isaac Lab v3.0.0 '
       'and v2.3.2 Python files (parsed with ast, executed against stub classes with the fields of the cited source) '
       'and the camera\'s pose in its parent frame; also orientation within 10⁻⁹ rad in each release\'s quaternion '
       'order, focal length and OpenCV intrinsics within 10⁻⁹ relative, the randomisation bounds equal to '
       'core/tolerance.bounds, perturbed_offset within 10⁻⁹ m and 10⁻⁹ rad of the same draw applied by '
       'core/tolerance.apply_draw, and sample_offset within the bounds; the three samples with calibrated lenses',
       'core/camrender.optical_pose taken to the parent link by RobotModel.link_pose; core/tolerance and core/factors '
       'for the draws; the Isaac Lab v3.0.0 and v2.3.2 source for the field names', 1e-9, 'm',
       refs=('isaaclab_camera_v300', 'isaaclab_camera_v232'), cost='study')
def _isaac_config(ctx):
    from ..core.robot.samples import sample_project
    from . import formats_isaac_lerobot as V
    rng = ctx.rng('formats.isaac_config')
    worst = {'pose_m': 0.0, 'pose_rad': 0.0, 'intrinsics_rel': 0.0, 'dr_m': 0.0, 'dr_rad': 0.0}
    detail, bad = {}, []
    for name in EXPORT_B_SAMPLES:
        d = V.isaac_config_errors(V.lens_variety(sample_project(name)), os.path.join(ctx.tmp(), f'isaac_{name}'),
                                  rng)
        for k in worst:
            worst[k] = max(worst[k], d[k])
        bad += [f'{name}: {m}' for m in d['mismatches']]
        detail[name] = d
    detail.update(worst=worst, mismatches=bad)
    ok = worst['pose_rad'] <= 1e-9 and worst['intrinsics_rel'] <= 1e-9 and worst['dr_m'] <= 1e-9 and \
        worst['dr_rad'] <= 1e-9 and not bad
    return Outcome(max(worst['pose_m'], worst['dr_m']), detail, ok=ok,
                   note='The files write every number with 17 significant digits (Python\'s repr), so values read '
                        'back to the last bit and the differences are the rounding of composing the transforms in '
                        'another order (measured 2026-11-07: 1.4 × 10⁻¹⁶ m, 8.2 × 10⁻¹⁶ rad); 10⁻⁹ m and rad is '
                        'far below a mount or quaternion-order error (millimetres, degrees). Isaac Lab itself is not '
                        'installed here: '
                        'the stub classes refuse any field the v3.0.0 or v2.3.2 source does not define and require '
                        'the ones it leaves MISSING.')


@check('formats.lerobot_export', 'formats', 'LeRobot info.json and train configs agree with the reader and the source',
       'mismatches: the exported v2.1 and v3.0 info.json against the info.json of a dataset written with the same '
       'features (state names, camera keys and sizes) and read back by DexVantage\'s LeRobot reader (keys, dtypes, '
       'shapes, names, video sizes and rates, paths); train-config keys that are not fields of the pinned '
       'TrainPipelineConfig, DatasetConfig or policy config; feature types; ACT\'s temporal-ensembling rule; camera '
       'shapes against info.json; the ACT and Diffusion Policy YAML through PyYAML and the OpenVLA entry executed '
       'against stub enums; the UR5e sample with its action-chunk recommendation applied',
       'core/datasets/lerobot (write_lerobot and LeRobotDataset); the LeRobot source at commit d490e47 (field names); '
       'ACT, Diffusion Policy and OpenVLA sources at their pinned commits', 0, 'mismatches',
       refs=('lerobot_dataset_format', 'lerobot_train_config', 'act_repo', 'diffusion_policy_repo', 'openvla_oxe'),
       cost='study')
def _lerobot_export(ctx):
    from ..core import jobs
    from ..core.robot.samples import sample_project
    from ..core.studies import action_chunk_sync as AC
    from . import formats_isaac_lerobot as V
    p = sample_project('ur5e')
    applied = []
    try:
        applied = AC.apply_recommendations(p, jobs.run_study('action_chunk_sync', p.snapshot()))
    except jobs.StudyError:
        pass
    d = V.lerobot_errors(p, os.path.join(ctx.tmp(), 'lerobot_export'))
    d['detail']['recommendations_applied_to'] = applied
    return Outcome(len(d['mismatches']), {'mismatches': d['mismatches'], **d['detail']},
                   note='LeRobot, ACT, Diffusion Policy and OpenVLA are not installed here, so their own loaders are '
                        'not run: info.json is checked through DexVantage\'s reader (itself checked against the '
                        'format by formats.lerobot_roundtrip), and the configs against the field names of the '
                        'pinned sources. The codec fields are not compared (the writer uses H.264 where PyAV has no '
                        'AV1 encoder).')


@check('formats.report', 'formats', 'Report HTML parses and lists every study present',
       'mismatches when report.html, built over five studies of the UR5e sample with requirements on two of them, '
       'is parsed by Python\'s html.parser: a study result without its section and heading, a section without a '
       'result, a study neither reported nor listed as not run, no scorecard, no provenance block with the project\'s '
       'content hash, fewer figures than the sections that have one; and report.pdf without a study or the hash '
       'when read by pdftotext (poppler), where installed',
       'the report\'s own results document (report.json) and core.provenance.project_digest; Python\'s html.parser; '
       'poppler\'s pdftotext', 0, 'mismatches', refs=('html_living_standard', 'pdf_32000', 'poppler_pdftotext'),
       cost='study')
def _report(ctx):
    from ..core import requirements as RQ
    from ..core.robot.samples import sample_project
    from . import formats_isaac_lerobot as V
    p = sample_project('ur5e')
    p.requirements = [r for r in RQ.default_requirements() if r.study in V.REPORT_STUDIES]
    d = V.report_errors(p, os.path.join(ctx.tmp(), 'report'))
    note = '' if d['pdf'].get('parser') else ('pdftotext is not installed, so the PDF is checked for its header and '
                                              'trailer only. ')
    return Outcome(len(d['mismatches']), d, note=note + 'Studies that refuse on this project are listed under "Not '
                   'run" with their sentence, which the check accepts.')
