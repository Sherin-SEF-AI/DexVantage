"""LeRobot, ACT, Diffusion Policy and OpenVLA export (spec §9.2, §7.13, §9.4): the dataset's info.json features and
camera keys (LeRobot v2.1 and v3.0), a ready-to-run LeRobot train config per project policy with its cameras and the
action-chunk study's delay and ensembling recommendations, the camera config of the original ACT and Diffusion Policy
code, and OpenVLA's observation keys.

    @exporter('lerobot', 'LeRobot, ACT, Diffusion Policy and OpenVLA configs', ...)
    options: policy (a policy uid; default every policy of the project), repo_id (default 'local/<project>'),
             dataset_root (default None: LeRobot's own cache), fps (the dataset's frame rate; default the policy's
             control rate, else the fastest camera, else 30 fps, a default), device (default None: LeRobot picks)

Files (under out_dir/lerobot/; <p> is the policy's file name):

    v2.1/meta/info.json, v3.0/meta/info.json   info.json of a dataset recorded with the rig: features
                                     (observation.state, action, one video feature per camera, the default features),
                                     zero counters, LeRobot's default encoder fields; not stamped (a template to copy)
    cameras.json                     camera keys, sensors, imagers, sizes and frame rates (stamped)
    <p>/train_config.json            TrainPipelineConfig for lerobot-train --config_path (not stamped: draccus refuses
                                     unknown keys)
    <p>/train.sh                     the command
    <p>/delta_timestamps.json        the delay offset as LeRobotDataset delta_timestamps (stamped)
    <p>/make_dataset.py              a LeRobotDataset built with them
    act/task_config.yaml             ACT's TASK_CONFIGS entry and its temporal aggregation constant
    diffusion_policy/task.yaml       Diffusion Policy's task shape_meta and n_latency_steps
    openvla/oxe_dataset_config.py    OpenVLA's OXE_DATASET_CONFIGS entry (and .json)

Sources, each read on 2026-11-07:
- LeRobot, https://github.com/huggingface/lerobot at commit d490e472a5d1a207cfe0255caaaf74b6d66e4656 (the commit
  core/datasets/lerobot.py follows; v2.1 at tag v0.3.3): configs/train.py (TrainPipelineConfig: dataset, policy,
  output_dir, job_name; validate() refuses push_to_hub without policy.repo_id; from_pretrained reads train_config.json
  through draccus), configs/default.py (DatasetConfig repo_id, root), configs/policies.py (PreTrainedConfig
  input_features / output_features {key: {type, shape}}, device, push_to_hub, repo_id), configs/types.py
  (FeatureType STATE, VISUAL, ACTION), policies/act/configuration_act.py (ACTConfig chunk_size, n_action_steps,
  temporal_ensemble_coeff: "n_action_steps must be 1 when using temporal ensembling"; "all images having the same
  shape"; action_delta_indices = range(chunk_size)), policies/diffusion/configuration_diffusion.py (DiffusionConfig
  horizon, n_action_steps, resize_shape (H, W), crop_shape (H, W)), the register_subclass names of policies/pi0,
  pi0_fast, smolvla and groot ('pi0', 'pi0_fast', 'smolvla', 'groot'), datasets/factory.py (resolve_delta_timestamps:
  delta timestamps come from the policy's delta indices, so no train-config field shifts them),
  datasets/lerobot_dataset.py (LeRobotDataset(repo_id, root, delta_timestamps)), datasets/utils.py (DatasetInfo
  ignores unknown keys with a warning), configs/video.py (the default encoder: libsvtav1, yuv420p, g 2, crf 30,
  preset 12).
- ACT, https://github.com/tonyzhaozh/act at commit 742c753c0d4a5d87076c8f69e5628c79a8cc5488: constants.py
  (SIM_TASK_CONFIGS entries: dataset_dir, num_episodes, episode_len, camera_names), imitate_episodes.py (TASK_CONFIGS
  of aloha_scripts for real tasks; --temporal_agg; exp_weights = np.exp(-k * np.arange(...)) with k = 0.01).
- Diffusion Policy, https://github.com/real-stanford/diffusion_policy at commit
  5ba07ac6661db573af695b419a7947ecb704690f: diffusion_policy/config/task/real_pusht_image.yaml (image_shape,
  shape_meta obs {camera_k: {shape, type: rgb}, low_dim}, action), config/train_diffusion_unet_real_image_workspace.yaml
  (n_latency_steps: 0), dataset/real_pusht_image_dataset.py ("handle latency by dropping first n_latency_steps
  action").
- OpenVLA, https://github.com/openvla/openvla at commit c8f03f48af692657d3060c19588038c7220e9af9:
  prismatic/vla/datasets/rlds/oxe/configs.py (OXE_DATASET_CONFIGS: image_obs_keys primary / secondary / wrist,
  depth_obs_keys, state_obs_keys, StateEncoding, ActionEncoding).

**Recommendations (§7.13).** A policy's stored recommendations (Policy.recommendations, the action-chunk study's
Apply) go where each framework has a place for them: temporal_ensemble_m -> ACTConfig.temporal_ensemble_coeff (the
same convention: w_i = exp(−m·i), i = 0 the oldest; n_action_steps 1) and ACT's k; delay_offset_s -> the action's
delta timestamps start at δ instead of 0 (LeRobot: delta_timestamps.json and make_dataset.py, because its train
config has no field for it), and Diffusion Policy's n_latency_steps = δ in steps (it drops the first n actions of a
window, which is a whole-step δ ≥ 0). A negative δ has no place in Diffusion Policy's config (said in the notes).
Without stored recommendations, each framework's own defaults are written and labelled (ACT's k = 0.01 and
temporal ensembling as the policy entry states it; n_latency_steps 0).

**Cameras.** One video feature per camera or depth sensor the policy takes (Policy.cameras; every one when it names
none), its primary imager (the RGB imager of a depth camera) at the stream size; key 'observation.images.<sensor
name>' as core/sim/episodes.py names them (lower case, the parenthesised part left out). Depth streams are not added
(LeRobot's depth features are 12-bit videos DexVantage does not write).

Known limits: the lerobot package, ACT, Diffusion Policy and OpenVLA are not installed here, so the files are checked
against the cited sources by parsing (formats.lerobot_info: info.json read back through DexVantage's LeRobot reader
after a write with the same features; the train config's keys against the dataclass fields read from the pinned
source), not by running those trainers; state and action are the robots' actuated joints (an end-effector action
space is not written); the OpenVLA entry assumes an RLDS dataset whose observation keys are the camera names and
'joint_state'; the released OpenVLA model reads the primary image only; ACT supports one image shape for every
camera, and a rig whose cameras differ is said to need a resize.
"""
from __future__ import annotations

import json
import math
import re


from ..jobs import StudyError
from . import _common as C
from . import exporter

FOLDER = 'lerobot'
SCHEMA = 'dexvantage.lerobot_export/1'
LEROBOT_URL = 'https://github.com/huggingface/lerobot'
ACT_URL, ACT_COMMIT = 'https://github.com/tonyzhaozh/act', '742c753c0d4a5d87076c8f69e5628c79a8cc5488'
DP_URL, DP_COMMIT = 'https://github.com/real-stanford/diffusion_policy', '5ba07ac6661db573af695b419a7947ecb704690f'
OPENVLA_URL, OPENVLA_COMMIT = 'https://github.com/openvla/openvla', 'c8f03f48af692657d3060c19588038c7220e9af9'
SOURCE_READ = '2026-11-07'
DEFAULT_FPS = 30                    # a default of DexVantage when nothing states a rate
ACT_K_DEFAULT = 0.01                # ACT imitate_episodes.py, k = 0.01 (the official code's constant)
DP_N_LATENCY_DEFAULT = 0            # Diffusion Policy train_diffusion_unet_real_image_workspace.yaml
# library policy id -> LeRobot's registered policy type (register_subclass at the pinned commit)
LEROBOT_TYPES = {'act': 'act', 'diffusion-policy-cnn': 'diffusion', 'pi0': 'pi0', 'pi0-fast': 'pi0_fast',
                 'smolvla-base': 'smolvla', 'gr00t-n1-2b': 'groot'}
# TrainPipelineConfig / DatasetConfig / PreTrainedConfig / ACTConfig / DiffusionConfig fields this export writes,
# as read at the pinned commit (the check compares the written keys with these)
TRAIN_FIELDS = ('dataset', 'policy', 'output_dir', 'job_name')
DATASET_FIELDS = ('repo_id', 'root')
POLICY_FIELDS = ('type', 'input_features', 'output_features', 'device', 'push_to_hub', 'repo_id')
ACT_FIELDS = ('chunk_size', 'n_action_steps', 'temporal_ensemble_coeff')
DIFFUSION_FIELDS = ('horizon', 'n_action_steps', 'resize_shape', 'crop_shape')
FEATURE_TYPES = ('STATE', 'VISUAL', 'ACTION')
# LeRobot's default encoder (configs/video.py at the pinned commit; DexVantage's writer uses the same,
# core/datasets/lerobot.ENCODER_DEFAULTS); the codec name is the stream's (AV1 from libsvtav1)
VIDEO_DEFAULT = {'video.codec': 'av1', 'video.pix_fmt': 'yuv420p'}


GREEK = {'π': 'pi', 'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 'λ': 'lambda', 'μ': 'mu', 'σ': 'sigma'}


def _slug(text: str) -> str:
    t = ''.join(GREEK.get(c, c) for c in str(text or ''))
    return C.frame_name(t)


def camera_key(name: str, used: set) -> str:
    """'observation.images.<sensor name>' as core/sim/episodes.py names a sensor's camera (lower case, the
    parenthesised part left out, then a-z, 0-9 and _), made unique with _2, _3, ..."""
    base = re.sub(r'\([^)]*\)', '', str(name or '')).strip().lower()
    base = re.sub(r'[^a-z0-9]+', '_', base).strip('_') or 'camera'
    k, i = f'observation.images.{base}', 2
    while k in used:
        k, i = f'observation.images.{base}_{i}', i + 1
    used.add(k)
    return k


# ====================================================================== what the dataset holds
def state_names(project) -> tuple[list, list]:
    """(names, notes): the actuated joints of every robot (RobotModel.joint_names), prefixed with the robot's name
    when the project has more than one."""
    from ..robot.model import RobotModelError, get_model
    names, notes = [], []
    many = len(project.robots) > 1
    for r in project.robots:
        try:
            m = get_model(project, r.uid)
        except RobotModelError as e:
            notes.append(f'{r.name} is left out of the state: {e}')
            continue
        names += [(f'{_slug(r.name)}.{j}' if many else j) for j in m.joint_names]
    return names, notes


def rig_streams(project, sensors=None) -> list:
    """[{'key', 'sensor_uid', 'sensor', 'imager', 'role', 'on_link', 'parent', 'width', 'height', 'fps'}] for the
    primary imager of each camera or depth sensor in use (sensors: limit to these uids)."""
    from ..studies import motion_blur as MB
    q, _ = C.joint_state(project, {})
    cams = C.rig_cameras(project, q, imagers='primary')
    if sensors is not None:
        cams = [c for c in cams if c.sensor.uid in set(sensors)]
    try:
        fps = {r['key']: (r.get('fps'), tuple(r.get('stream') or ())) for r in MB.imaging(project)}
    except StudyError:
        fps = {}
    out, used = [], set()
    for rc in cams:
        f, mode = fps.get(f'{rc.sensor.uid}|{rc.imager.name}', (None, ()))
        if f and mode and tuple(mode) != (rc.camera.width, rc.camera.height):
            f = None
        out.append({'key': camera_key(rc.sensor.name, used), 'sensor_uid': rc.sensor.uid, 'sensor': rc.sensor.name,
                    'imager': rc.imager.name, 'role': rc.imager.role, 'on_link': bool(rc.sensor.mount.on_link),
                    'parent': rc.parent, 'width': rc.camera.width, 'height': rc.camera.height,
                    'fps': float(f) if f else None})
    return out


def info_json(version: str, fps: int, robot_type, names: list, streams: list) -> dict:
    """meta/info.json of an empty dataset with these features, as core/datasets/lerobot.write_lerobot writes it (the
    video info: LeRobot's default encoder, the stream's size and the dataset's fps)."""
    from ..datasets import lerobot as LR
    feats: dict = {}
    if names:
        feats[LR.STATE_KEY] = {'dtype': 'float32', 'shape': [len(names)], 'names': list(names)}
        feats[LR.ACTION_KEY] = {'dtype': 'float32', 'shape': [len(names)], 'names': list(names)}
    for s in sorted(streams, key=lambda x: x['key']):          # the writer's order (sorted camera keys)
        info = {'video.height': s['height'], 'video.width': s['width'], **VIDEO_DEFAULT}
        if version == 'v2.1':
            info['video.is_depth_map'] = False
        info['video.fps'] = int(fps)
        info['video.channels'] = 3
        info['has_audio'] = False
        if version == 'v3.0':
            info.update({'video.g': LR.ENCODER_DEFAULTS['g'], 'video.crf': LR.ENCODER_DEFAULTS['crf'],
                         'video.preset': LR.ENCODER_DEFAULTS['preset_svtav1_v30'], 'video.fast_decode': 0,
                         'is_depth_map': False})
        feats[s['key']] = {'dtype': 'video', 'shape': [s['height'], s['width'], 3],
                           'names': ['height', 'width', 'channels'], 'info': info}
    feats.update({k: dict(v) for k, v in LR.DEFAULT_FEATURES.items()})
    if version == 'v2.1':
        return {'codebase_version': 'v2.1', 'robot_type': robot_type, 'total_episodes': 0, 'total_frames': 0,
                'total_tasks': 0, 'total_videos': 0, 'total_chunks': 0, 'chunks_size': LR.DEFAULT_CHUNK_SIZE,
                'fps': int(fps), 'splits': {}, 'data_path': LR.V21_DATA_PATH,
                'video_path': LR.V21_VIDEO_PATH if streams else None, 'features': feats}
    return {'codebase_version': 'v3.0', 'fps': int(fps), 'features': feats, 'total_episodes': 0, 'total_frames': 0,
            'total_tasks': 0, 'chunks_size': LR.DEFAULT_CHUNK_SIZE,
            'data_files_size_in_mb': LR.DEFAULT_DATA_FILE_SIZE_IN_MB,
            'video_files_size_in_mb': LR.DEFAULT_VIDEO_FILE_SIZE_IN_MB, 'data_path': LR.V30_DATA_PATH,
            'video_path': LR.V30_VIDEO_PATH if streams else None, 'robot_type': robot_type, 'splits': {}}


# ====================================================================== per policy
def policy_stem(pol, used: set) -> str:
    """'lerobot/<library id>_<policy name>' (file-name safe, unique)."""
    name = _slug(pol.name)
    lib = _slug(pol.library_id) if pol.library_id else ''
    base = f'{lib}_{name}' if lib and not name.startswith(lib) else (name or lib or 'policy')
    stem, k = f'{FOLDER}/{base}', 2
    while stem in used:
        stem, k = f'{FOLDER}/{base}_{k}', k + 1
    used.add(stem)
    return stem


def _rec(pol, key):
    r = (pol.recommendations or {}).get(key)
    if isinstance(r, dict) and r.get('value') is not None:
        return r
    return None


def _num(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else v


def policy_plan(project, pol, streams_all: list, names: list, options: dict) -> dict:
    """What the policy's train config holds, with where each figure comes from."""
    from .. import policies as PO
    try:
        fields = PO.resolve(project, pol.uid)
    except StudyError:
        fields = {}
    val = {k: v[0] for k, v in fields.items()}
    src = {k: v[1] for k, v in fields.items()}
    uids = {s.uid for s in PO.policy_cameras(project, pol)}
    streams = [s for s in streams_all if s['sensor_uid'] in uids]
    ltype = LEROBOT_TYPES.get(pol.library_id)
    notes = []
    fps = options.get('fps')
    fps_src = 'given with the export'
    if fps is None and _num(val.get('control_hz')):
        fps, fps_src = val['control_hz'], f'the policy\'s control rate ({src.get("control_hz")})'
    if fps is None:
        fs = [s['fps'] for s in streams if s['fps']]
        if fs:
            fps, fps_src = max(fs), 'the fastest camera\'s stream'
    if fps is None:
        fps, fps_src = DEFAULT_FPS, f'{DEFAULT_FPS} fps, a default (nothing states a rate)'
    fps = float(fps)
    delay = _rec(pol, 'delay_offset_s')
    ens = _rec(pol, 'temporal_ensemble_m')
    d_s = float(delay['value']) if delay else 0.0
    steps = int(round(d_s * fps))
    if delay and abs(steps / fps - d_s) > 1e-9:
        notes.append(f'{pol.name}: the delay offset {d_s * 1000:.4g} ms is {d_s * fps:.3g} frames at {fps:g} fps; '
                     f'it is written as {steps} whole frames ({steps / fps * 1000:.4g} ms), since delta timestamps '
                     'are multiples of 1/fps.')
    chunk = _num(val.get('action_chunk'))
    m_entry = val.get('temporal_ensemble_m')
    m = float(ens['value']) if ens else (float(m_entry) if isinstance(m_entry, (int, float)) and
                                         not isinstance(m_entry, bool) else None)
    m_src = (ens.get('source') if ens else
             (f'the policy entry ({src.get("temporal_ensemble_m")})' if m is not None else 'not used'))
    shapes = {(s['height'], s['width']) for s in streams}
    if ltype in ('act', 'diffusion') and len(shapes) > 1:
        notes.append(f'{pol.name}: LeRobot\'s {ltype} supports one image shape for every camera ("Right now we only '
                     'support all images having the same shape"); these cameras differ ('
                     + ', '.join(f'{w} × {h}' for h, w in sorted(shapes)) + '): record them at one size or resize.')
    return {'policy': pol, 'type': ltype, 'streams': streams, 'fps': fps, 'fps_source': fps_src,
            'delay_s': steps / fps if delay else 0.0, 'delay_steps': steps if delay else 0,
            'delay_source': delay.get('source') if delay else 'none stored (0, LeRobot\'s delta timestamps start at 0)',
            'm': m, 'm_source': m_src, 'chunk': int(chunk) if chunk else None,
            'execute': _num(val.get('execute_steps')), 'pre_resize': val.get('pre_resize'),
            'input_resolution': val.get('input_resolution'), 'crop_rule': val.get('crop_rule'),
            'names': names, 'notes': notes}


def train_config(plan: dict, options: dict, project) -> dict:
    """The TrainPipelineConfig document (only fields read at the pinned commit; module docstring)."""
    pol = plan['policy']
    feats = {}
    if plan['names']:
        feats['observation.state'] = {'type': 'STATE', 'shape': [len(plan['names'])]}
    for s in plan['streams']:
        feats[s['key']] = {'type': 'VISUAL', 'shape': [3, s['height'], s['width']]}
    out_feats = {'action': {'type': 'ACTION', 'shape': [len(plan['names'])]}} if plan['names'] else {}
    p = {'type': plan['type'], 'input_features': feats, 'output_features': out_feats,
         'device': options.get('device'), 'push_to_hub': False, 'repo_id': None}
    if plan['type'] == 'act':
        if plan['chunk']:
            p['chunk_size'] = plan['chunk']
        if plan['m'] is not None:
            p['temporal_ensemble_coeff'] = float(plan['m'])
            p['n_action_steps'] = 1
        elif plan['execute'] and plan['chunk']:
            p['n_action_steps'] = int(min(plan['execute'], plan['chunk']))
    elif plan['type'] == 'diffusion':
        if plan['chunk']:
            p['horizon'] = plan['chunk']
        if plan['execute']:
            p['n_action_steps'] = int(plan['execute'])
        pre, inp = plan['pre_resize'], plan['input_resolution']
        if isinstance(pre, (list, tuple)) and len(pre) == 2:
            p['resize_shape'] = [int(pre[1]), int(pre[0])]
            if isinstance(inp, (list, tuple)) and len(inp) == 2 and plan['crop_rule'] == 'center_crop':
                p['crop_shape'] = [int(inp[1]), int(inp[0])]
    job = _slug(f'{project.name}_{pol.name}')
    return {'dataset': {'repo_id': str(options.get('repo_id') or f'local/{_slug(project.name)}'),
                        'root': options.get('dataset_root')},
            'policy': p, 'output_dir': f'outputs/train/{job}', 'job_name': job}


def delta_timestamps(plan: dict) -> dict:
    """{'action': [δ + k / fps]} over the policy's action window (ACT: range(chunk_size); Diffusion Policy:
    range(1 − n_obs_steps, 1 − n_obs_steps + horizon) with its default n_obs_steps 2)."""
    fps, d = plan['fps'], plan['delay_steps']
    n = plan['chunk'] or 1
    if plan['type'] == 'diffusion':
        idx = list(range(1 - 2, 1 - 2 + n))
    else:
        idx = list(range(n))
    return {'action': [(d + i) / fps for i in idx]}


MAKE_DATASET = '''"""A LeRobotDataset whose actions start {delay_ms} ms after each observation (the action-chunk
study's delay offset for {policy}, {steps} frame(s) at {fps:g} fps). LeRobot's train config has no field for this
(its delta timestamps come from the policy's delta indices, datasets/factory.py at commit d490e47), so build the
dataset here, or shift the action column of the dataset by the same frames.
"""
import json
import pathlib

from lerobot.datasets.lerobot_dataset import LeRobotDataset

DELTA = json.loads((pathlib.Path(__file__).parent / "delta_timestamps.json").read_text())["delta_timestamps"]

dataset = LeRobotDataset({repo_id!r}, root={root!r}, delta_timestamps=DELTA)
'''


def _yaml(doc, header: list) -> str:
    import yaml
    return ''.join(f'# {h}\n' if h else '#\n' for h in header) + yaml.safe_dump(doc, sort_keys=False,
                                                                                 allow_unicode=True, width=110)


def _roles(streams: list) -> dict:
    """OpenVLA's image roles: primary = the first static camera, secondary = the next, wrist = the first on a link."""
    static = [s for s in streams if not s['on_link']]
    wrist = [s for s in streams if s['on_link']]
    base = lambda s: s['key'].replace('observation.images.', '') if s else None    # noqa: E731
    return {'primary': base(static[0]) if static else (base(wrist[1]) if len(wrist) > 1 else None),
            'secondary': base(static[1]) if len(static) > 1 else None,
            'wrist': base(wrist[0]) if wrist else None}


# ====================================================================== the exporter
@exporter('lerobot', 'LeRobot, ACT, Diffusion Policy and OpenVLA configs', needs_robot=True,
          description='LeRobot v2.1 and v3.0 info.json features and camera keys, a train config per policy with the '
                      'action-chunk recommendations, ACT and Diffusion Policy camera YAML, OpenVLA observation keys.')
def export_lerobot(project, out_dir: str, options: dict, ctx) -> dict:
    from .. import policies as PO
    o = dict(options or {})
    if o.get('fps') is not None:
        try:
            f = float(o['fps'])
        except (TypeError, ValueError):
            f = float('nan')
        if not math.isfinite(f) or f <= 0 or abs(f - round(f)) > 1e-9:
            raise StudyError('The dataset\'s frame rate must be a whole number of frames per second above zero '
                             '(LeRobot stores it as one).')
        o['fps'] = int(round(f))
    names, notes = state_names(project)
    streams = rig_streams(project)
    C.need_cameras(streams, 'The LeRobot export')
    pols = list(project.policies)
    if o.get('policy'):
        pols = [p for p in pols if p.uid == o['policy']]
        if not pols:
            raise StudyError('The project has no policy with that id.')
    robot_type = _slug(project.primary_robot.name) if project.primary_robot else None
    files, unstamped = [], []
    plans = [policy_plan(project, p, streams, names, o) for p in pols]
    ds_fps = o.get('fps') or (int(round(plans[0]['fps'])) if plans else None)
    if ds_fps is None:
        fs = [s['fps'] for s in streams if s['fps']]
        ds_fps = int(round(max(fs))) if fs else DEFAULT_FPS
        if not fs:
            notes.append(f'No camera states its frame rate and no policy gives a control rate: the dataset\'s fps '
                         f'is {DEFAULT_FPS}, a default.')
    slow = sorted({s['fps'] for s in streams if s['fps'] and s['fps'] < ds_fps - 1e-9})
    if slow:
        notes.append(f'The dataset\'s rate ({ds_fps} fps) is above the cameras\' ({", ".join(f"{f:g}" for f in slow)} '
                     'fps): LeRobot stores one frame per row, so camera frames repeat; record at the camera rate or '
                     'give fps.')
    if plans and abs(plans[0]['fps'] - round(plans[0]['fps'])) > 1e-9:
        notes.append(f'LeRobot stores a whole-number fps; {plans[0]["fps"]:g} is written as {ds_fps}.')
    for v in ('v2.1', 'v3.0'):
        rel = f'{FOLDER}/{v}/meta/info.json'
        files.append(C.write_json(out_dir, rel, info_json(v, ds_fps, robot_type, names, streams)))
        unstamped.append(rel)
    notes.append('info.json is a template of a dataset recorded with this rig (zero counters; the video fields are '
                 'LeRobot\'s default encoder: AV1, yuv420p, g 2, crf 30); LeRobot rewrites the video fields from the '
                 'files it encodes. It is not stamped with provenance (the manifest holds its hash).')
    files.append(C.write_json(out_dir, f'{FOLDER}/cameras.json', {
        'schema': SCHEMA, 'fps': ds_fps, 'state_names': names, 'cameras': streams,
        'source': {'lerobot': {'url': LEROBOT_URL, 'commit': _lr().SOURCE_COMMIT, 'v21_commit':
                               _lr().SOURCE_COMMIT_V21, 'read': SOURCE_READ}}}))
    if not pols:
        notes.append('The project has no policy, so no train config is written: add one on the Policies page (its '
                     'cameras, chunk length and the action-chunk study\'s recommendations go into the config).')
    stems: set = set()
    for plan in plans:
        pol = plan['policy']
        notes += plan['notes']
        stem = policy_stem(pol, stems)
        if plan['type'] is None:
            notes.append(f'{pol.name} ({PO.title_of(pol)}) has no LeRobot policy type at the pinned commit, so it '
                         'gets no train config; its cameras are in cameras.json and the OpenVLA keys.')
            continue
        if not plan['streams']:
            notes.append(f'{pol.name} takes no camera that is in use, so it gets no train config.')
            continue
        tc = train_config(plan, o, project)
        rel = f'{stem}/train_config.json'
        files.append(C.write_json(out_dir, rel, tc))
        unstamped.append(rel)
        files.append(C.write_text(out_dir, f'{stem}/train.sh', (
            '#!/bin/sh\n'
            f'# Train {pol.name} on the dataset {tc["dataset"]["repo_id"]} with the cameras DexVantage planned.\n'
            f'# LeRobot at commit {_lr().SOURCE_COMMIT[:7]} ({LEROBOT_URL}): lerobot-train reads train_config.json\n'
            '# through TrainPipelineConfig.from_pretrained; flags after it override (e.g. --steps=20000).\n'
            '# The delay offset is not a train-config field: see make_dataset.py.\n'
            'cd "$(dirname "$0")" && exec lerobot-train --config_path=train_config.json "$@"\n')))
        dt = delta_timestamps(plan)
        files.append(C.write_json(out_dir, f'{stem}/delta_timestamps.json', {
            'schema': 'dexvantage.delta_timestamps/1', 'policy': pol.name, 'fps': plan['fps'],
            'delay_offset_s': plan['delay_s'], 'delay_steps': plan['delay_steps'],
            'delay_source': plan['delay_source'], 'delta_timestamps': dt,
            'meaning': 'the action paired with the observation at t is the one stamped t + delay_offset_s (+ k/fps '
                       'over the window); LeRobot\'s own default starts the window at 0'}))
        files.append(C.write_text(out_dir, f'{stem}/make_dataset.py', MAKE_DATASET.format(
            delay_ms=f'{plan["delay_s"] * 1000:.4g}', policy=pol.name, steps=plan['delay_steps'], fps=plan['fps'],
            repo_id=tc['dataset']['repo_id'], root=tc['dataset']['root'])))
        if plan['type'] == 'act':
            notes.append(f'{pol.name}: temporal_ensemble_coeff ' + (f'{plan["m"]:g} ({plan["m_source"]})'
                                                                     if plan['m'] is not None else
                                                                     'not set (the policy entry uses none)') + '.')
        if plan['delay_steps']:
            notes.append(f'{pol.name}: delay offset {plan["delay_steps"]} frame(s) ({plan["delay_source"]}): '
                         'delta_timestamps.json and make_dataset.py apply it (LeRobot\'s train config has no field '
                         'for it).')
    # the original ACT and Diffusion Policy code, OpenVLA
    act_plan = next((p for p in plans if p['type'] == 'act'), None)
    dp_plan = next((p for p in plans if p['type'] == 'diffusion'), None)
    act_streams = act_plan['streams'] if act_plan else streams
    k = act_plan['m'] if act_plan and act_plan['m'] is not None else ACT_K_DEFAULT
    act_cams = [s['key'].replace('observation.images.', '') for s in act_streams]
    act_doc = {_slug(project.name): {'dataset_dir': None, 'num_episodes': None, 'episode_len': None,
                                     'camera_names': act_cams},
               'temporal_agg': True, 'temporal_agg_k': float(k),
               'chunk_size': act_plan['chunk'] if act_plan and act_plan['chunk'] else None,
               'delay_offset_steps': act_plan['delay_steps'] if act_plan else 0}
    files.append(C.write_text(out_dir, 'act/task_config.yaml', _yaml(act_doc, [
        f'ACT ({ACT_URL} at commit {ACT_COMMIT[:7]}, read {SOURCE_READ}): imitate_episodes.py reads',
        'TASK_CONFIGS[task] = {dataset_dir, num_episodes, episode_len, camera_names} (constants.py); paste the entry',
        'below into it and fill dataset_dir, num_episodes and episode_len. camera_names are the image keys of your',
        'HDF5 episodes (/observations/images/<name>).',
        f'temporal_agg_k: imitate_episodes.py hard-codes k = {ACT_K_DEFAULT} in exp(-k * arange(n)) (line 255);',
        '  set it to this value there. ' + (f'It is the action-chunk study\'s m ({act_plan["m_source"]}).'
                                            if act_plan and act_plan['m'] is not None and
                                            _rec(act_plan['policy'], 'temporal_ensemble_m') else
                                            'It is the code\'s own k (no recommendation stored).'),
        'delay_offset_steps: ACT has no setting for it; shift the actions by this many steps when you build the',
        '  episodes (0: none stored).'])))
    dp_streams = dp_plan['streams'] if dp_plan else streams
    pre = dp_plan['pre_resize'] if dp_plan else None
    shape_meta = {'obs': {}, 'action': {'shape': [len(names)]}}
    for s in dp_streams:
        h, w = (int(pre[1]), int(pre[0])) if isinstance(pre, (list, tuple)) and len(pre) == 2 else (s['height'],
                                                                                                   s['width'])
        shape_meta['obs'][s['key'].replace('observation.images.', '')] = {'shape': [3, h, w], 'type': 'rgb'}
    shape_meta['obs']['robot_joint_pos'] = {'shape': [len(names)], 'type': 'low_dim'}
    n_lat = dp_plan['delay_steps'] if dp_plan else DP_N_LATENCY_DEFAULT
    if dp_plan and n_lat < 0:
        notes.append(f'{dp_plan["policy"].name}: the delay offset is negative ({n_lat} steps); Diffusion Policy\'s '
                     'n_latency_steps drops actions and cannot shift them the other way, so it is written as 0.')
        n_lat = 0
    dp_doc = {'name': _slug(project.name), 'shape_meta': shape_meta, 'n_latency_steps': int(n_lat),
              'horizon': dp_plan['chunk'] if dp_plan and dp_plan['chunk'] else None,
              'n_action_steps': int(dp_plan['execute']) if dp_plan and dp_plan['execute'] else None}
    files.append(C.write_text(out_dir, 'diffusion_policy/task.yaml', _yaml(dp_doc, [
        f'Diffusion Policy ({DP_URL} at commit {DP_COMMIT[:7]}, read {SOURCE_READ}): the task\'s shape_meta as',
        'diffusion_policy/config/task/real_pusht_image.yaml writes it (obs: rgb cameras [3, H, W] and low_dim state;',
        'action), and the workspace\'s n_latency_steps (train_diffusion_unet_real_image_workspace.yaml; the dataset',
        '"handle[s] latency by dropping first n_latency_steps action"). '
        + ('The image size is the policy entry\'s pre_resize.' if pre else 'The image size is the camera stream.'),
        'n_latency_steps: ' + (f'the action-chunk study\'s delay offset in steps ({dp_plan["delay_source"]}).'
                               if dp_plan and _rec(dp_plan['policy'], 'delay_offset_s') else
                               f'{DP_N_LATENCY_DEFAULT}, the workspace default (no recommendation stored).'),
        'robot_joint_pos names the low-dim key of your recorded state; rename it to match your dataset.'])))
    roles = _roles(streams)
    ov = {'image_obs_keys': roles, 'depth_obs_keys': {'primary': None, 'secondary': None, 'wrist': None},
          'state_obs_keys': ['joint_state'], 'state_encoding': 'StateEncoding.JOINT',
          'action_encoding': 'ActionEncoding.JOINT_POS'}
    ds_name = f'dexvantage_{_slug(project.name)}'
    py = ('"""OpenVLA observation keys for a dataset recorded with the DexVantage rig "' + project.name + '".\n\n'
          'An OXE_DATASET_CONFIGS entry as prismatic/vla/datasets/rlds/oxe/configs.py defines them\n'
          f'({OPENVLA_URL} at commit {OPENVLA_COMMIT[:7]}, read {SOURCE_READ}): image_obs_keys primary (external\n'
          'RGB), secondary, wrist; depth_obs_keys; state_obs_keys; the state and action encodings. The keys are the\n'
          'RLDS observation keys of\n'
          'the cameras (DexVantage names them after the sensors); OpenVLA as released reads the primary image only.\n'
          'That file defines StateEncoding.JOINT as "Joint Angles (7, <PAD> if fewer) + Gripper Open/Close (1)" and\n'
          'ActionEncoding.JOINT_POS as "Joint Delta Position (7) + Gripper Open/Close (1)": a dataset whose action is\n'
          'absolute joint positions needs its actions turned into deltas, and a two-arm rig is JOINT_BIMANUAL /\n'
          'JOINT_POS_BIMANUAL (6 joints + gripper per arm).\n'
          'Paste the entry into OXE_DATASET_CONFIGS (this file imports the two enums from that module).\n"""\n'
          'from prismatic.vla.datasets.rlds.oxe.configs import ActionEncoding, StateEncoding\n\n'
          f'{ds_name.upper()} = {{\n'
          f'    "{ds_name}": {{\n'
          f'        "image_obs_keys": {json.dumps(roles).replace("null", "None")},\n'
          '        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},\n'
          '        "state_obs_keys": ["joint_state"],\n'
          '        "state_encoding": StateEncoding.JOINT,\n'
          '        "action_encoding": ActionEncoding.JOINT_POS,\n'
          '    },\n}\n')
    files.append(C.write_text(out_dir, 'openvla/oxe_dataset_config.py', py))
    files.append(C.write_json(out_dir, 'openvla/oxe_dataset_config.json', {
        'schema': 'dexvantage.openvla_keys/1', 'dataset': ds_name, 'entry': ov,
        'source': {'url': OPENVLA_URL, 'commit': OPENVLA_COMMIT, 'file': 'prismatic/vla/datasets/rlds/oxe/configs.py',
                   'read': SOURCE_READ}}))
    notes.append('OpenVLA: the entry uses StateEncoding.JOINT and ActionEncoding.JOINT_POS, which OpenVLA\'s '
                 'configs.py defines as joint angles and joint DELTA positions (7 joints, padded, + gripper); '
                 'convert absolute joint actions to deltas before training'
                 + ('; this project has more than one robot, for which that file has JOINT_BIMANUAL and '
                    'JOINT_POS_BIMANUAL (6 joints + gripper per arm): edit the entry if the dataset is bimanual.'
                    if len(project.robots) > 1 else '.'))
    if roles['primary'] is None:
        notes.append('OpenVLA: no static camera is in use, so its primary image (an external RGB view) has no camera; '
                     'add a third-person camera.')
    return {'files': files, 'notes': notes, 'unstamped': unstamped}


def _lr():
    from ..datasets import lerobot as LR
    return LR


__all__ = ['export_lerobot', 'info_json', 'rig_streams', 'state_names', 'train_config', 'policy_plan',
           'delta_timestamps', 'camera_key']
