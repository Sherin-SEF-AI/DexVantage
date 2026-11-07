"""The round trips of the Isaac, LeRobot and report exporters (core/exporters/isaac.py, lerobot.py, report.py), used by
the formats.isaac_usd, formats.isaac_config, formats.lerobot_export and formats.report checks in checks_formats.py.

- USD: the exported camera prims, written by pxr and by the text writer, re-opened with pxr (usd-core); every camera
  prim's world transform, focal length and apertures, its OpenCV lens attributes, and points projected through pxr's
  own Gf.Camera frustum against DexVantage's camera model.
- Isaac Lab configs: the generated v3.0.0 and v2.3.2 Python files parsed (ast) and executed in a namespace whose only
  imports are math, random and stub isaaclab modules (stub classes that accept only the fields the cited Isaac Lab
  source defines and require the ones it leaves MISSING), then every value compared with an independent computation
  of the camera (core/camrender.optical_pose taken back to the link by RobotModel.link_pose), the tolerance bounds,
  and perturbed_offset against core/tolerance.apply_draw moving the sensor through core/factors.
- LeRobot: the exported info.json features re-created by a write of matching episodes (core/datasets/lerobot
  write_lerobot) and read back through DexVantage's LeRobot reader; the train config's keys against the dataclass
  fields of the pinned LeRobot source; the ACT and Diffusion Policy YAML through PyYAML; the OpenVLA entry executed
  against stub enums.
- Report: report.html parsed with Python's html.parser (every study present is a section with its heading, the
  scorecard and the provenance block with the project's content hash are there); report.pdf read by poppler's
  pdftotext when it is installed.

The stub field lists are transcribed from the sources the exporters cite (read 2026-11-07): isaac-sim/IsaacLab tag
v3.0.0 (commit e3f7d9f) sensors/camera/camera_cfg.py, sensors/sensor_base_cfg.py, sim/spawners/sensors/sensors_cfg.py,
sim/spawners/spawner_cfg.py; tag v2.3.2 (commit 37ddf62) the same files; huggingface/lerobot commit d490e47
configs/train.py, configs/default.py, configs/policies.py, configs/types.py, policies/act/configuration_act.py,
policies/diffusion/configuration_diffusion.py and the pi0, pi0_fast, smolvla and groot configurations.
"""
from __future__ import annotations

import ast
import builtins
import json
import math
import os
import random
import types

import numpy as np

# ====================================================================== Isaac Lab stubs (fields of the cited source)
SENSOR_BASE = ('class_type', 'cloning_contexts', 'prim_path', 'update_period', 'debug_vis')
CAMERA_V3 = SENSOR_BASE + ('offset', 'spawn', 'depth_clipping_behavior', 'data_types', 'width', 'height',
                           'update_latest_camera_pose', 'semantic_filter', 'colorize_semantic_segmentation',
                           'colorize_instance_id_segmentation', 'colorize_instance_segmentation',
                           'semantic_segmentation_mapping', 'background_color', 'renderer_cfg', 'isp_cfg')
CAMERA_V2 = SENSOR_BASE + ('offset', 'spawn', 'depth_clipping_behavior', 'data_types', 'width', 'height',
                           'update_latest_camera_pose', 'semantic_filter', 'colorize_semantic_segmentation',
                           'colorize_instance_id_segmentation', 'colorize_instance_segmentation',
                           'semantic_segmentation_mapping')
OFFSET = ('pos', 'rot', 'convention')
SPAWNER = ('func', 'visible', 'semantic_tags', 'copy_from_source')
PINHOLE_V3 = SPAWNER + ('projection_type', 'distortion', 'clipping_range', 'focal_length', 'focus_distance', 'f_stop',
                        'horizontal_aperture', 'vertical_aperture', 'horizontal_aperture_offset',
                        'vertical_aperture_offset', 'lock_camera')
PINHOLE_V2 = SPAWNER + ('projection_type', 'clipping_range', 'focal_length', 'focus_distance', 'f_stop',
                        'horizontal_aperture', 'vertical_aperture', 'horizontal_aperture_offset',
                        'vertical_aperture_offset', 'lock_camera')
OPENCV_BASE = ('model', 'fx', 'fy', 'cx', 'cy', 'image_size', 'apply_lens_distortion')
OPENCV_PINHOLE = OPENCV_BASE + ('k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'p1', 'p2', 's1', 's2', 's3', 's4')
OPENCV_FISHEYE = OPENCV_BASE + ('k1', 'k2', 'k3', 'k4', 'max_fov')
REQUIRED_CAMERA = ('prim_path', 'width', 'height', 'spawn')     # MISSING in the source
BROWN_NAMES = ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6')
KB_NAMES = ('k1', 'k2', 'k3', 'k4')


class _Cfg:
    FIELDS: tuple = ()
    REQUIRED: tuple = ()
    DEFAULTS: dict = {}

    def __init__(self, **kw):
        bad = sorted(set(kw) - set(self.FIELDS))
        if bad:
            raise TypeError(f'{type(self).__name__} has no field {", ".join(bad)}')
        miss = [k for k in self.REQUIRED if k not in kw]
        if miss:
            raise TypeError(f'{type(self).__name__} needs {", ".join(miss)}')
        self.__dict__.update(dict(self.DEFAULTS, **kw))

    def replace(self, **kw):
        return type(self)(**{**{k: v for k, v in self.__dict__.items() if k in self.FIELDS}, **kw})


def isaac_stubs(v3: bool) -> dict:
    """{module name: module} of stub isaaclab modules with the fields of the cited release."""
    def cls(name, fields, required=(), defaults=None):
        return type(name, (_Cfg,), {'FIELDS': tuple(fields), 'REQUIRED': tuple(required),
                                    'DEFAULTS': dict(defaults or {})})
    Offset = cls('OffsetCfg', OFFSET, defaults={'pos': (0.0, 0.0, 0.0),
                                                'rot': (0.0, 0.0, 0.0, 1.0) if v3 else (1.0, 0.0, 0.0, 0.0),
                                                'convention': 'ros'})
    Cam = cls('CameraCfg', CAMERA_V3 if v3 else CAMERA_V2, REQUIRED_CAMERA,
              {'offset': Offset(), 'data_types': ['rgb'], 'update_period': 0.0})
    Cam.OffsetCfg = Offset
    sim = types.ModuleType('isaaclab.sim')
    sim.PinholeCameraCfg = cls('PinholeCameraCfg', PINHOLE_V3 if v3 else PINHOLE_V2,
                               defaults={'horizontal_aperture': 20.955, 'focal_length': 24.0,
                                         'clipping_range': (0.01, 1e6), 'distortion': None})
    if v3:
        sim.OpenCvPinholeDistortionCfg = cls('OpenCvPinholeDistortionCfg', OPENCV_PINHOLE,
                                             ('fx', 'fy', 'cx', 'cy', 'image_size'), {'model': 'opencvPinhole'})
        sim.OpenCvFisheyeDistortionCfg = cls('OpenCvFisheyeDistortionCfg', OPENCV_FISHEYE,
                                             ('fx', 'fy', 'cx', 'cy', 'image_size'), {'model': 'opencvFisheye'})
    sensors = types.ModuleType('isaaclab.sensors')
    if v3:
        sensors.CameraCfg = Cam
    else:
        sensors.TiledCameraCfg = Cam
    root = types.ModuleType('isaaclab')
    root.sim, root.sensors = sim, sensors
    return {'isaaclab': root, 'isaaclab.sim': sim, 'isaaclab.sensors': sensors}


def sandbox_exec(text: str, modules: dict, allowed=('math', 'random'), name: str = 'exported.py') -> dict:
    """Execute text (after ast.parse) with builtins whose __import__ serves only `modules` and the allowed standard
    modules, and without open, exec, eval, compile and input; returns the namespace."""
    ast.parse(text)
    real = {m: __import__(m) for m in allowed}

    def imp(mod, globals=None, locals=None, fromlist=(), level=0):
        if mod in modules:
            return modules[mod] if fromlist else modules[mod.split('.')[0]]
        if mod in real:
            return real[mod]
        raise ImportError(f'the sandbox does not provide {mod}')
    b = {k: getattr(builtins, k) for k in dir(builtins)
         if not k.startswith('_') and k not in ('open', 'exec', 'eval', 'compile', 'input', 'breakpoint')}
    b['__import__'] = imp
    b['__build_class__'] = builtins.__build_class__
    ns = {'__builtins__': b, '__name__': 'dexvantage_sandbox'}
    exec(compile(text, name, 'exec'), ns)        # noqa: S102 — a file DexVantage wrote, in a restricted namespace
    return ns


# ====================================================================== independent camera poses
def parent_optical(project, sensor, imager, q) -> np.ndarray:
    """T_parent_optical by another route than the exporter: the imager's world pose (core/camrender.optical_pose)
    taken back into its parent frame by the robot's own link pose (RobotModel.link_pose)."""
    from ..core.camrender import optical_pose
    from ..core.robot.model import get_model
    Tw = optical_pose(project, sensor, imager, q).matrix()
    if not sensor.mount.on_link:
        return Tw
    m = get_model(project, sensor.mount.robot_uid)
    Tl = np.asarray(m.link_pose(q.get(sensor.mount.robot_uid, m.home), sensor.mount.parent_frame), float)
    return np.linalg.inv(Tl) @ Tw


def _quat_mat(q, order: str) -> np.ndarray:
    from ..core.frames import quat_to_matrix
    q = list(q)
    if order == 'wxyz':
        q = q[1:] + q[:1]
    return quat_to_matrix(np.asarray(q, float))


def _rot_err_rad(Ra, Rb) -> float:
    """The angle of Ra^T Rb, as atan2(|sin|, cos) from its skew part and trace (accurate near 0, where acos of the
    trace alone loses about 2 × 10⁻⁸ rad)."""
    D = np.asarray(Ra, float).T @ np.asarray(Rb, float)
    s = 0.5 * np.linalg.norm([D[2, 1] - D[1, 2], D[0, 2] - D[2, 0], D[1, 0] - D[0, 1]])
    c = (np.trace(D) - 1) / 2
    return float(math.atan2(s, c))


def _imagers(project) -> dict:
    """{(sensor uid, imager name): (sensor, imager)} of the enabled cameras and depth sensors."""
    from ..core import library as LB
    from ..core.studies import coverage as CV
    out = {}
    for s in project.variant.sensors:
        if s.enabled and s.kind in ('camera', 'depth'):
            for im in LB.sensor_imagers(s, CV.sensor_mode(s, {})):
                out[(s.uid, im.name)] = (s, im)
    return out


def _home(project) -> dict:
    from ..core.robot.model import RobotModelError, get_model
    q = {}
    for r in project.robots:
        try:
            q[r.uid] = np.asarray(get_model(project, r.uid).home, float)
        except RobotModelError:
            pass
    return q


def lens_variety(project):
    """A copy of project whose first two cameras carry a unit calibration: a Brown-Conrady lens with an off-centre
    principal point and non-square pixels, and a Kannala-Brandt lens (so every lens path of the exporter runs)."""
    p = project.snapshot()
    cams = [s for s in p.variant.sensors if s.enabled and s.kind in ('camera', 'depth')]
    if cams:
        cams[0].calibration = {'K': [[600.0, 0.0, 300.25], [0.0, 640.0, 260.5], [0.0, 0.0, 1.0]],
                               'D': [0.1, -0.05, 0.001, 0.002, 0.01], 'width': 640, 'height': 480, 'model': 'brown'}
    if len(cams) > 1:
        cams[1].calibration = {'K': [[400.0, 0.0, 330.0], [0.0, 401.0, 230.0], [0.0, 0.0, 1.0]],
                               'D': [0.02, -0.01, 0.003, -0.001], 'width': 640, 'height': 480, 'model': 'kb'}
    return p


# ====================================================================== Isaac: USD
def isaac_usd_errors(project, folder: str) -> dict:
    """Write the camera prims with pxr and with the text writer, re-open both with pxr, compare (module docstring).
    {'proj_px', 'pose_m', 'pose_rad', 'lens_rel', 'mismatches', 'cameras'}."""
    from pxr import Usd, UsdGeom

    from ..core.camrender import optical_pose
    from ..core.exporters import isaac as I
    from ..core.optics import CameraModel
    recs, _ = I.camera_records(project)
    q = _home(project)
    ims = _imagers(project)
    os.makedirs(folder, exist_ok=True)
    paths = {'pxr': os.path.join(folder, 'pxr.usda'), 'text': os.path.join(folder, 'text.usda')}
    if os.path.exists(paths['pxr']):
        os.remove(paths['pxr'])
    I.write_usd_pxr(paths['pxr'], recs, project.name)
    with open(paths['text'], 'w', encoding='utf-8') as fh:
        fh.write(I.usda_text(recs, project.name))
    out = {'proj_px': 0.0, 'pose_m': 0.0, 'pose_rad': 0.0, 'lens_rel': 0.0, 'mismatches': [], 'cameras': len(recs)}
    rng = np.random.default_rng(7)
    for kind, path in paths.items():
        st = Usd.Stage.Open(path)
        if st is None:
            out['mismatches'].append(f'{kind}: pxr cannot open the file')
            continue
        if UsdGeom.GetStageMetersPerUnit(st) != 1.0 or UsdGeom.GetStageUpAxis(st) != UsdGeom.Tokens.z:
            out['mismatches'].append(f'{kind}: stage units or up axis')
        found = sorted(p.GetName() for p in st.Traverse() if p.IsA(UsdGeom.Camera))
        if found != sorted(r['name'] for r in recs):
            out['mismatches'].append(f'{kind}: camera prims {found}')
        for r in recs:
            prim = st.GetPrimAtPath(f'/{I.ROOT}/{r["parent_prim"]}/{r["name"]}')
            if not prim or not prim.IsA(UsdGeom.Camera):
                out['mismatches'].append(f'{kind}: no camera prim {r["name"]}')
                continue
            s, im = ims[(r['sensor_uid'], r['imager'])]
            cam = im.camera
            T_ref = optical_pose(project, s, im, q).matrix()
            Tw = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
            Tw[:3, :3] = Tw[:3, :3] @ np.diag([1.0, -1.0, -1.0])
            out['pose_m'] = max(out['pose_m'], float(np.abs(Tw[:3, 3] - T_ref[:3, 3]).max()))
            out['pose_rad'] = max(out['pose_rad'], _rot_err_rad(Tw[:3, :3], T_ref[:3, :3]))
            uc = UsdGeom.Camera(prim)
            W, H = cam.width, cam.height
            f = uc.GetFocalLengthAttr().Get()
            ha, va = uc.GetHorizontalApertureAttr().Get(), uc.GetVerticalApertureAttr().Get()
            out['lens_rel'] = max(out['lens_rel'], abs(W * f / ha - cam.fx) / cam.fx, abs(H * f / va - cam.fy) / cam.fy)
            fr = uc.GetCamera(Usd.TimeCode.Default()).frustum
            V, P = np.array(fr.ComputeViewMatrix()), np.array(fr.ComputeProjectionMatrix())
            Pc = np.c_[rng.uniform(-0.25, 0.25, (40, 2)), rng.uniform(0.3, 2.0, (40, 1))]
            Pw = Pc @ T_ref[:3, :3].T + T_ref[:3, 3]
            h = np.c_[Pw, np.ones(len(Pw))] @ V @ P
            ndc = h[:, :2] / h[:, 3:4]
            uv_usd = np.c_[(ndc[:, 0] + 1) / 2 * W - 0.5, (1 - ndc[:, 1]) / 2 * H - 0.5]
            uv, ok = CameraModel(W, H, cam.fx, cam.fy, cam.cx, cam.cy).project(Pc)
            out['proj_px'] = max(out['proj_px'], float(np.abs(uv_usd[ok] - uv[ok]).max()))
            model = prim.GetAttribute('omni:lensdistortion:model').Get()
            want = r['distortion']
            if (model or None) != (want['model'] if want else None):
                out['mismatches'].append(f'{kind}: {r["name"]} lens model {model}')
                continue
            if not want:
                continue
            pre = f'omni:lensdistortion:{model}'
            for k in ('fx', 'fy', 'cx', 'cy'):
                v, ref = prim.GetAttribute(f'{pre}:{k}').Get(), getattr(cam, k)
                if v is None or abs(v - ref) > 1e-6 * max(1.0, abs(ref)):        # float32 attributes
                    out['mismatches'].append(f'{kind}: {r["name"]} {k} = {v} (want {ref})')
            size = prim.GetAttribute(f'{pre}:imageSize').Get()
            if size is None or (int(size[0]), int(size[1])) != (W, H):
                out['mismatches'].append(f'{kind}: {r["name"]} imageSize {size}')
            if cam.model in ('brown', 'kb'):
                for n_, ref in zip(BROWN_NAMES if cam.model == 'brown' else KB_NAMES, cam.dist):
                    v = prim.GetAttribute(f'{pre}:{n_}').Get()
                    if v is None or abs(v - ref) > 1e-6 * max(1.0, abs(ref)):
                        out['mismatches'].append(f'{kind}: {r["name"]} {n_} = {v} (want {ref})')
            if I.DISTORTION_API[model] not in [str(x) for x in prim.GetAppliedSchemas()] and \
                    I.DISTORTION_API[model] not in str(prim.GetMetadata('apiSchemas')):
                out['mismatches'].append(f'{kind}: {r["name"]} without the {I.DISTORTION_API[model]} schema')
    return out


# ====================================================================== Isaac: Python configs
def isaac_config_errors(project, folder: str, rng) -> dict:
    """Export, execute both config files against the stubs, compare (module docstring)."""
    from ..core import tolerance as TL
    from ..core.exporters import export
    from ..core.exporters import isaac as I
    man = export(project, folder, ['isaac'])
    if man['formats']['isaac']['status'] != 'ok':
        return {'mismatches': [f'refused: {man["formats"]["isaac"].get("reason")}'], 'pose_m': 0.0,
                'pose_rad': 0.0, 'intrinsics_rel': 0.0, 'dr_m': 0.0, 'dr_rad': 0.0, 'cameras': 0}
    recs, _ = I.camera_records(project)
    q = _home(project)
    ims = _imagers(project)
    bounds = TL.bounds(project)
    out = {'pose_m': 0.0, 'pose_rad': 0.0, 'intrinsics_rel': 0.0, 'dr_m': 0.0, 'dr_rad': 0.0, 'draws_checked': 0,
           'mismatches': [], 'cameras': len(recs), 'files': {}}
    for v3, rel in ((True, I.V3_FILE), (False, I.V2_FILE)):
        with open(os.path.join(folder, rel), encoding='utf-8') as fh:
            text = fh.read()
        try:
            ns = sandbox_exec(text, isaac_stubs(v3), name=os.path.basename(rel))
        except Exception as e:                           # noqa: BLE001 — reported as a mismatch
            out['mismatches'].append(f'{rel}: {type(e).__name__}: {e}')
            continue
        order = 'xyzw' if v3 else 'wxyz'
        cams = ns.get('CAMERAS') or {}
        if set(cams) != {r['name'] for r in recs}:
            out['mismatches'].append(f'{rel}: cameras {sorted(cams)}')
        out['files'][rel] = len(cams)
        for r in recs:
            cfg = cams.get(r['name'])
            if cfg is None:
                continue
            s, im = ims[(r['sensor_uid'], r['imager'])]
            cam = im.camera
            T_ref = parent_optical(project, s, im, q)
            out['pose_m'] = max(out['pose_m'], float(np.abs(np.asarray(cfg.offset.pos, float) - T_ref[:3, 3]).max()))
            out['pose_rad'] = max(out['pose_rad'], _rot_err_rad(_quat_mat(cfg.offset.rot, order), T_ref[:3, :3]))
            if cfg.offset.convention != 'ros':
                out['mismatches'].append(f'{rel}: {r["name"]} convention {cfg.offset.convention}')
            if (cfg.width, cfg.height) != (cam.width, cam.height):
                out['mismatches'].append(f'{rel}: {r["name"]} size {cfg.width} × {cfg.height}')
            sp = cfg.spawn
            out['intrinsics_rel'] = max(out['intrinsics_rel'],
                                        abs(cfg.width * sp.focal_length / sp.horizontal_aperture - cam.fx) / cam.fx)
            path = cfg.prim_path
            if not path.startswith('{ENV_REGEX_NS}/Robot/' if s.mount.on_link else '{ENV_REGEX_NS}/') or (
                    s.mount.on_link and f'/{I.prim_token(s.mount.parent_frame)}/' not in path):
                out['mismatches'].append(f'{rel}: {r["name"]} prim path {path}')
            if v3:
                d = sp.distortion
                needs = cam.model != 'pinhole' or abs(cam.fx - cam.fy) > 1e-9 * cam.fx or \
                    abs(cam.cx - (cam.width - 1) / 2) > 1e-6 or abs(cam.cy - (cam.height - 1) / 2) > 1e-6
                if needs != (d is not None):
                    out['mismatches'].append(f'{rel}: {r["name"]} distortion {d}')
                if d is not None:
                    for k in ('fx', 'fy', 'cx', 'cy'):
                        out['intrinsics_rel'] = max(out['intrinsics_rel'],
                                                    abs(getattr(d, k) - getattr(cam, k)) / cam.fx)
                    if tuple(d.image_size) != (cam.width, cam.height):
                        out['mismatches'].append(f'{rel}: {r["name"]} image_size {d.image_size}')
                    want = 'opencvPinhole' if cam.model in ('brown', 'pinhole') else 'opencvFisheye'
                    if d.model != want:
                        out['mismatches'].append(f'{rel}: {r["name"]} model {d.model}')
                    if cam.model in ('brown', 'kb'):
                        for n_, ref in zip(BROWN_NAMES if cam.model == 'brown' else KB_NAMES, cam.dist):
                            if abs(getattr(d, n_, 0.0) - ref) > 1e-15 * max(1.0, abs(ref)):
                                out['mismatches'].append(f'{rel}: {r["name"]} {n_}')
            b = bounds.get(s.uid)
            dr = ns['DR'][r['name']]['bounds']
            if b is None or dr is None:
                if (b is None) != (dr is None):
                    out['mismatches'].append(f'{rel}: {r["name"]} bounds {dr}')
                continue
            if any(abs(x - y) > 1e-15 for a_, b_ in zip(dr['position_m'], b['position_m']) for x, y in zip(a_, b_)) \
                    or any(abs(x - y) > 1e-12 for a_, b_ in zip(dr['rotation_deg'], b['rotation_deg'])
                           for x, y in zip(a_, b_)):
                out['mismatches'].append(f'{rel}: {r["name"]} bounds differ from core/tolerance.bounds')
            for _ in range(2):
                d6 = [float(rng.uniform(lo, hi)) for lo, hi in b['position_m']] + \
                     [float(rng.uniform(lo, hi)) for lo, hi in b['rotation_deg']]
                p2, r2 = ns['perturbed_offset'](r['name'], d6)
                moved = TL.apply_draw(project, {s.uid: d6})
                s2 = next(x for x in moved.variant.sensors if x.uid == s.uid)
                T2 = parent_optical(moved, s2, im, q)
                out['dr_m'] = max(out['dr_m'], float(np.abs(np.asarray(p2) - T2[:3, 3]).max()))
                out['dr_rad'] = max(out['dr_rad'], _rot_err_rad(_quat_mat(r2, order), T2[:3, :3]))
                out['draws_checked'] += 1
            pr = random.Random(int(rng.integers(1 << 30)))
            lim = [hi for _lo, hi in b['position_m']] + [hi for _lo, hi in b['rotation_deg']]
            for _ in range(20):
                dd = ns['sample_offset'](r['name'], pr)[2]
                if any(abs(x) > h_ + 1e-15 for x, h_ in zip(dd, lim)):
                    out['mismatches'].append(f'{rel}: {r["name"]} a draw outside its bounds')
                    break
        try:
            if set(ns['randomized_cameras'](1)) != set(cams):
                out['mismatches'].append(f'{rel}: randomized_cameras')
        except Exception as e:                           # noqa: BLE001 — reported
            out['mismatches'].append(f'{rel}: randomized_cameras {type(e).__name__}: {e}')
    return out


# ====================================================================== LeRobot
TRAIN_PIPELINE = ('dataset', 'env', 'policy', 'reward_model', 'output_dir', 'job_name', 'resume', 'seed',
                  'cudnn_deterministic', 'tf32', 'num_workers', 'batch_size', 'prefetch_factor', 'persistent_workers',
                  'dataloader_multiprocessing_context', 'steps', 'env_eval_freq', 'log_freq', 'eval_steps',
                  'max_eval_samples', 'tolerance_s', 'save_checkpoint', 'save_freq', 'checkpoint_format',
                  'use_policy_training_preset', 'optimizer', 'scheduler', 'parallelism', 'accelerator', 'eval', 'ema',
                  'trackio', 'peft', 'job', 'save_checkpoint_to_hub', 'sample_weighting', 'rename_map')
DATASET_CONFIG = ('repo_id', 'repo_type', 'root', 'episodes', 'exclude_episodes', 'image_transforms', 'revision',
                  'use_imagenet_stats', 'video_backend', 'return_uint8', 'depth_output_unit', 'streaming')
PRETRAINED = ('type', 'n_obs_steps', 'input_features', 'output_features', 'device', 'dtype', 'use_amp', 'use_peft',
              'push_to_hub', 'repo_id', 'private', 'tags', 'license', 'pretrained_path', 'pretrained_revision')
POLICY_EXTRA = {'act': ('chunk_size', 'n_action_steps', 'temporal_ensemble_coeff', 'normalization_mapping',
                        'vision_backbone', 'dropout', 'kl_weight'),
                'diffusion': ('horizon', 'n_action_steps', 'resize_shape', 'crop_ratio', 'crop_shape',
                              'crop_is_random', 'drop_n_last_frames'),
                'pi0': ('chunk_size', 'n_action_steps'), 'pi0_fast': ('chunk_size', 'n_action_steps'),
                'smolvla': ('chunk_size', 'n_action_steps'), 'groot': ('chunk_size', 'n_action_steps')}
FEATURE_TYPES = ('STATE', 'VISUAL', 'ENV', 'ACTION', 'REWARD', 'LANGUAGE')


def _episodes_for(info: dict, n: int = 2):
    """Episodes whose features are those of info.json (state names, camera keys and sizes)."""
    from ..core.datasets import episode as E
    feats = info['features']
    names = (feats.get('observation.state') or {}).get('names') or []
    fps = float(info['fps'])
    t = np.arange(n) / fps
    q = np.linspace(0.0, 0.1, n * max(1, len(names))).reshape(n, -1) if names else np.zeros((n, 0))
    cams = {}
    for k, ft in feats.items():
        if ft.get('dtype') != 'video':
            continue
        h, w = ft['shape'][0], ft['shape'][1]
        frames = [np.full((h, w, 3), 40 * (i + 1), np.uint8) for i in range(n)]

        def dec(idx, frames=frames):
            for i in (range(len(frames)) if idx is None else idx):
                yield i, frames[i]
        cams[k] = E.CameraStream(k, w, h, fps, t.copy(), _decode=dec)
    return [E.Episode('e0', 0, t, q, list(names), q.copy(), list(names), fps, info.get('robot_type'), 'task', cams,
                      {})]


def lerobot_errors(project, folder: str) -> dict:
    """Export; for each version write episodes with the exported features and read them back; compare the features;
    check the train configs' keys, the YAML files and the OpenVLA entry. {'mismatches', 'detail'}."""
    import yaml

    from ..core.datasets import lerobot as LR
    from ..core.exporters import export
    from ..core.exporters import lerobot as XL
    man = export(project, folder, ['lerobot'])
    row = man['formats']['lerobot']
    if row['status'] != 'ok':
        return {'mismatches': [f'refused: {row.get("reason")}'], 'detail': {}}
    bad, detail = [], {'files': row['files']}
    for v in LR.SUPPORTED:
        with open(os.path.join(folder, XL.FOLDER, v, 'meta', 'info.json'), encoding='utf-8') as fh:
            info = json.load(fh)
        if 'provenance' in info:
            bad.append(f'{v}: info.json carries a provenance block (LeRobot warns on unknown keys)')
        root = os.path.join(folder, f'roundtrip_{v}')
        written = LR.write_lerobot(root, _episodes_for(info), v, info['fps'], info.get('robot_type'))
        back = LR.LeRobotDataset(root).info
        for k in ('codebase_version', 'fps', 'data_path', 'video_path', 'robot_type', 'chunks_size'):
            if back.get(k) != info.get(k):
                bad.append(f'{v}: {k} {info.get(k)!r} -> {back.get(k)!r}')
        if set(back) != set(info):
            bad.append(f'{v}: top-level keys differ: {sorted(set(back) ^ set(info))}')
        fe, fb = info['features'], back['features']
        if list(fe) != list(fb):
            bad.append(f'{v}: features {list(fe)} -> {list(fb)}')
        for k in set(fe) & set(fb):
            for a in ('dtype', 'shape', 'names'):
                if fe[k].get(a) != fb[k].get(a):
                    bad.append(f'{v}: {k}.{a} {fe[k].get(a)!r} -> {fb[k].get(a)!r}')
            if fe[k].get('dtype') == 'video':
                ie, ib = fe[k].get('info') or {}, fb[k].get('info') or {}
                if set(ie) != set(ib):
                    bad.append(f'{v}: {k}.info keys {sorted(set(ie) ^ set(ib))}')
                for a in ('video.height', 'video.width', 'video.channels', 'video.fps', 'video.pix_fmt'):
                    if ie.get(a) != ib.get(a):
                        bad.append(f'{v}: {k}.info {a} {ie.get(a)!r} -> {ib.get(a)!r}')
        detail[v] = {'features': list(fe), 'codec_written': written.get('codec')}
    with open(os.path.join(folder, XL.FOLDER, 'v3.0', 'meta', 'info.json'), encoding='utf-8') as fh:
        feats = json.load(fh)['features']
    nconf = 0
    for rel in row['files']:
        if not rel.endswith('train_config.json'):
            continue
        nconf += 1
        with open(os.path.join(folder, rel), encoding='utf-8') as fh:
            tc = json.load(fh)
        bad += [f'{rel}: {k}' for k in tc if k not in TRAIN_PIPELINE]
        bad += [f'{rel}: dataset.{k}' for k in tc.get('dataset', {}) if k not in DATASET_CONFIG]
        if not tc.get('dataset', {}).get('repo_id'):
            bad.append(f'{rel}: dataset.repo_id is required')
        pol = tc.get('policy') or {}
        extra = POLICY_EXTRA.get(pol.get('type'))
        if extra is None:
            bad.append(f'{rel}: policy type {pol.get("type")}')
            continue
        bad += [f'{rel}: policy.{k}' for k in pol if k not in PRETRAINED + extra]
        for grp in ('input_features', 'output_features'):
            for k, ft in (pol.get(grp) or {}).items():
                if ft.get('type') not in FEATURE_TYPES or not isinstance(ft.get('shape'), list):
                    bad.append(f'{rel}: {grp}.{k}')
        if pol.get('push_to_hub', True) and not pol.get('repo_id'):
            bad.append(f'{rel}: push_to_hub without repo_id (TrainPipelineConfig.validate refuses it)')
        if pol.get('type') == 'act':
            if pol.get('temporal_ensemble_coeff') is not None and pol.get('n_action_steps', 100) > 1:
                bad.append(f'{rel}: temporal ensembling needs n_action_steps 1')
            if pol.get('n_action_steps', 0) > pol.get('chunk_size', 100):
                bad.append(f'{rel}: n_action_steps above chunk_size')
        for k, ft in (pol.get('input_features') or {}).items():
            if ft.get('type') == 'VISUAL':
                f = feats.get(k)
                if f is None or [3, f['shape'][0], f['shape'][1]] != ft['shape']:
                    bad.append(f'{rel}: {k} is not a camera of info.json at {ft["shape"]}')
            elif ft.get('type') == 'STATE' and ft['shape'] != feats.get('observation.state', {}).get('shape'):
                bad.append(f'{rel}: {k} shape')
    detail['train_configs'] = nconf
    with open(os.path.join(folder, 'act', 'task_config.yaml'), encoding='utf-8') as fh:
        act = yaml.safe_load(fh)
    entry = next((v for v in act.values() if isinstance(v, dict) and 'camera_names' in v), None)
    if entry is None or not set(entry) >= {'dataset_dir', 'num_episodes', 'episode_len', 'camera_names'}:
        bad.append('act/task_config.yaml: the TASK_CONFIGS entry')
    with open(os.path.join(folder, 'diffusion_policy', 'task.yaml'), encoding='utf-8') as fh:
        dp = yaml.safe_load(fh)
    sm = dp.get('shape_meta') or {}
    if not sm.get('obs') or 'action' not in sm or not isinstance(dp.get('n_latency_steps'), int):
        bad.append('diffusion_policy/task.yaml: shape_meta or n_latency_steps')
    for k, o in (sm.get('obs') or {}).items():
        if o.get('type') not in ('rgb', 'low_dim') or len(o.get('shape') or []) not in (1, 3):
            bad.append(f'diffusion_policy/task.yaml: obs {k}')
    enums = types.ModuleType('prismatic.vla.datasets.rlds.oxe.configs')
    enums.StateEncoding = types.SimpleNamespace(NONE=-1, POS_EULER=1, POS_QUAT=2, JOINT=3, JOINT_BIMANUAL=4)
    enums.ActionEncoding = types.SimpleNamespace(EEF_POS=1, JOINT_POS=2, JOINT_POS_BIMANUAL=3, EEF_R6=4)
    with open(os.path.join(folder, 'openvla', 'oxe_dataset_config.py'), encoding='utf-8') as fh:
        ns = sandbox_exec(fh.read(), {'prismatic.vla.datasets.rlds.oxe.configs': enums}, (),
                          'oxe_dataset_config.py')
    entries = [v for k, v in ns.items() if k.isupper() and isinstance(v, dict)]
    ent = next(iter(entries[0].values())) if entries else {}
    if set(ent) != {'image_obs_keys', 'depth_obs_keys', 'state_obs_keys', 'state_encoding', 'action_encoding'} or \
            set(ent.get('image_obs_keys') or {}) != {'primary', 'secondary', 'wrist'}:
        bad.append('openvla/oxe_dataset_config.py: the OXE entry')
    return {'mismatches': bad, 'detail': detail}


# ====================================================================== report
REPORT_STUDIES = ('bandwidth', 'lighting', 'motion_blur', 'time_sync', 'action_chunk_sync')


def report_errors(project, folder: str, studies=REPORT_STUDIES) -> dict:
    """Export the report over the given studies and parse it back. {'mismatches', 'sections', 'figures', 'pdf'}."""
    import shutil
    import subprocess
    from html.parser import HTMLParser

    from ..core import jobs
    from ..core import provenance as PV
    from ..core.exporters import export
    from ..core.exporters import report as XR

    man = export(project, folder, ['report'], {'report': {'studies': list(studies)}})
    row = man['formats']['report']
    if row['status'] != 'ok':
        return {'mismatches': [f'refused: {row.get("reason")}'], 'sections': 0, 'figures': 0, 'pdf': {}}
    with open(os.path.join(folder, XR.JSON_FILE), encoding='utf-8') as fh:
        doc = json.load(fh)
    present = [s['id'] for s in doc['sections']]

    class _Page(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.studies, self.ids, self.svg, self.h2, self.text, self._h2 = [], set(), 0, [], [], False

        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            if a.get('data-study'):
                self.studies.append(a['data-study'])
            if a.get('id'):
                self.ids.add(a['id'])
            self.svg += tag == 'svg'
            if tag == 'h2':
                self._h2 = True
                self.h2.append('')

        def handle_endtag(self, tag):
            if tag == 'h2':
                self._h2 = False

        def handle_data(self, data):
            self.text.append(data)
            if self._h2:
                self.h2[-1] += data

    page = _Page()
    with open(os.path.join(folder, XR.HTML_FILE), encoding='utf-8') as fh:
        page.feed(fh.read())
    page.close()
    bad = []
    if page.studies != present:
        bad.append(f'sections {page.studies} against results {present}')
    for sid in present:
        if jobs.STUDIES[sid].title not in page.h2:
            bad.append(f'no heading for {sid}')
    for sid in studies:
        if sid not in present and sid not in [r['id'] for r in doc['refused']]:
            bad.append(f'{sid} neither reported nor listed as not run')
    if 'provenance' not in page.ids:
        bad.append('no provenance block')
    sha = PV.project_digest(project)['sha256']
    if sha not in ''.join(page.text):
        bad.append('the provenance block lacks the project content hash')
    if project.requirements and 'scorecard' not in page.ids:
        bad.append('no scorecard')
    nfig = sum(1 for s in doc['sections'] if s.get('figure'))
    if page.svg < nfig:
        bad.append(f'{page.svg} figures for {nfig}')
    pdf = {'parser': None}
    path = os.path.join(folder, XR.PDF_FILE)
    with open(path, 'rb') as fh:
        data = fh.read()
    if not data.startswith(b'%PDF-') or not data.rstrip().endswith(b'%%EOF'):
        bad.append('the PDF header or trailer')
    tool = shutil.which('pdftotext')
    if tool:
        r = subprocess.run([tool, path, '-'], capture_output=True, text=True, timeout=60)
        pdf = {'parser': 'pdftotext', 'returncode': r.returncode, 'stderr': r.stderr.strip()[:200]}
        txt = ' '.join(r.stdout.split())
        if r.returncode != 0 or r.stderr.strip():
            bad.append(f'pdftotext: {r.returncode} {r.stderr.strip()[:120]}')
        for sid in present:
            if ' '.join(jobs.STUDIES[sid].title.split()) not in txt:
                bad.append(f'the PDF lacks {sid}')
        if sha not in txt.replace(' ', ''):
            bad.append('the PDF lacks the project content hash')
    return {'mismatches': bad, 'sections': len(present), 'figures': page.svg, 'pdf': pdf, 'studies': present}


def _usda_blocks(text: str) -> dict:
    """{camera name: {attr: value text, 'matrix': 4 × 4 or None, 'parent_matrix': 4 × 4 or None}} from USD text, by a
    small reader of the exporter's layout (def Xform / def Camera blocks with brace nesting)."""
    import re
    out, stack = {}, []

    def mat(s):
        nums = [float(x) for x in re.findall(r'[-+0-9.eE]+', s)]
        return np.asarray(nums, float).reshape(4, 4).T if len(nums) == 16 else None
    lines = text.splitlines()
    pending, in_meta = None, False
    for ln in lines:
        if in_meta:                                  # the prim's metadata ( ... ): customData braces are not blocks
            in_meta = ln.strip() != ')'
            continue
        m = re.match(r'\s*def (Xform|Camera) "([^"]+)"', ln)
        if m:
            pending = (m.group(1), m.group(2))
            in_meta = ln.rstrip().endswith('(')
            continue
        if ln.strip() == '{':
            stack.append({'kind': pending[0] if pending else 'meta', 'name': pending[1] if pending else '',
                          'attrs': {}})
            pending = None
            continue
        if ln.strip() == '}' and stack:
            node = stack.pop()
            if node['kind'] == 'Camera':
                parent = next((n for n in reversed(stack) if n['kind'] == 'Xform' and 'matrix' in n['attrs']), None)
                node['attrs']['parent_matrix'] = parent['attrs']['matrix'] if parent else None
                out[node['name']] = node['attrs']
            continue
        if stack:
            a = re.match(r'\s*(?:custom |uniform )*[\w\[\]0-9]+ ([\w:]+) = (.+)$', ln)
            if a:
                key, val = a.group(1), a.group(2)
                if key == 'xformOp:transform':
                    stack[-1]['attrs']['matrix'] = mat(val)
                else:
                    stack[-1]['attrs'][key] = val
    return out


def isaac_usd_text_errors(project, folder: str) -> dict:
    """Without pxr: the text-written file read back by _usda_blocks; the pinhole the attributes define (GfCamera's
    frustum written out again: fx = W f / hAp, cx = (W − 1)/2 − W hOff / hAp, fy = H f / vAp,
    cy = (H − 1)/2 + H vOff / vAp) and the camera's world transform against the reference."""
    from ..core.camrender import optical_pose
    from ..core.exporters import isaac as I
    recs, _ = I.camera_records(project)
    q = _home(project)
    ims = _imagers(project)
    blocks = _usda_blocks(I.usda_text(recs, project.name))
    out = {'proj_px': 0.0, 'pose_m': 0.0, 'pose_rad': 0.0, 'lens_rel': 0.0, 'mismatches': [], 'cameras': len(recs),
           'parser': 'text'}
    for r in recs:
        b = blocks.get(r['name'])
        if b is None or b.get('matrix') is None:
            out['mismatches'].append(f'text: no camera {r["name"]}')
            continue
        s, im = ims[(r['sensor_uid'], r['imager'])]
        cam = im.camera
        T = b['matrix'] if b.get('parent_matrix') is None else b['parent_matrix'] @ b['matrix']
        T = T.copy()
        T[:3, :3] = T[:3, :3] @ np.diag([1.0, -1.0, -1.0])
        T_ref = optical_pose(project, s, im, q).matrix()
        out['pose_m'] = max(out['pose_m'], float(np.abs(T[:3, 3] - T_ref[:3, 3]).max()))
        out['pose_rad'] = max(out['pose_rad'], _rot_err_rad(T[:3, :3], T_ref[:3, :3]))
        f, ha, va = (float(b[k]) for k in ('focalLength', 'horizontalAperture', 'verticalAperture'))
        ho, vo = float(b['horizontalApertureOffset']), float(b['verticalApertureOffset'])
        W, H = cam.width, cam.height
        got = (W * f / ha, H * f / va, (W - 1) / 2 - W * ho / ha, (H - 1) / 2 + H * vo / va)
        out['proj_px'] = max(out['proj_px'], max(abs(a - b_) for a, b_ in zip(got, (cam.fx, cam.fy, cam.cx, cam.cy))))
    return out


def selftest_isaac_lerobot(folder: str) -> tuple[bool, str]:
    """The self-test line: the SO-100 sample's Isaac Lab configs executed against the stubs and compared, its USD
    re-read (pxr when installed, else the text reader), and its LeRobot info.json, train config and YAML checked
    (formats.isaac_*, formats.lerobot_export run the full comparisons)."""
    import time

    from ..core.robot.samples import sample_project
    t0 = time.perf_counter()
    p = lens_variety(sample_project('so100'))
    c = isaac_config_errors(p, os.path.join(folder, 'isaac'), np.random.default_rng(3))
    from ..core.exporters import isaac as I
    u = isaac_usd_errors(p, os.path.join(folder, 'usd')) if I.have_pxr() else isaac_usd_text_errors(p, '')
    lr = lerobot_errors(sample_project('so100'), os.path.join(folder, 'lerobot'))
    bad = c['mismatches'] + u['mismatches'] + lr['mismatches']
    ms = (time.perf_counter() - t0) * 1000
    if bad or c['pose_m'] > 1e-9 or c['dr_m'] > 1e-9 or u['proj_px'] > 1e-3:
        return False, (bad[0] if bad else f'Isaac pose {c["pose_m"]:.3g} m, USD {u["proj_px"]:.3g} px')
    return True, (f'{c["cameras"]} Isaac cameras exact, USD {u["proj_px"]:.1g} px, LeRobot info.json and '
                  f'{lr["detail"].get("train_configs", 0)} train config ({ms:.0f} ms)')
