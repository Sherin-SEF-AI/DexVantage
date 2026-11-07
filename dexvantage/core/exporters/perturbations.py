"""Perturbed camera extrinsics for augmentation (spec §8.5): n draws of every enabled sensor's mount from its
tolerance or calibration covariance (core/tolerance.py), as JSON and a NumPy .npz, for Isaac Lab domain
randomisation (§9.2), the 3DGS renderer (§9.3) and DexVantage's own MuJoCo renders (core/sim/render
extrinsic_delta).

    @exporter('perturbations', 'Perturbed extrinsics (augmentation and domain randomisation)')
    options: n (draws, default 100), seed (default 0), scale (default 1), distribution ('uniform' default or
             'normal'), q ({robot uid: joint vector} for the world poses; default each robot's home)

Files (under out_dir/perturbations/):

    perturbations.json   {'schema': 'dexvantage.perturbations/1', 'n', 'seed', 'scale', 'distribution', 'sources',
                          'conventions', 'sensors': [{'uid', 'name', 'parent_frame', 'robot_uid', 'imager', 'width',
                          'height', 'fx', 'fy', 'cx', 'cy'}], 'draws': [[per sensor {'d', 'delta_body',
                          'mount': {'position', 'quaternion_xyzw'}, 'isaac_lab': {'pos', 'rot_xyzw', 'rot_wxyz',
                          'convention': 'ros'}, 'gs': {'position', 'rotation'}}]]}
    perturbations.npz    d (n, S, 6), delta_body, T_parent_body, T_parent_optical, T_world_optical (n, S, 4, 4),
                         sensor_uids, sensor_names (S,), seed, scale, distribution (scalars)

Conventions (read from the targets' own source, 2026-11-07):
- Isaac Lab CameraCfg.OffsetCfg: pos, rot "w.r.t. the parent frame" with convention "ros" (forward +Z, up −Y: the
  REP-103 optical frame written here). Its quaternion order changed between releases: (w, x, y, z) in v2.3.0 and
  (x, y, z, w) in v3.0.0 and main (source/isaaclab/isaaclab/sensors/camera/camera_cfg.py,
  https://github.com/isaac-sim/IsaacLab, main at commit d4be263); both orders are written, named. The parent here is
  the sensor's mount frame (a robot link, or the world for a static mount): the camera prim must be attached there.
- 3DGS (graphdeco-inria/gaussian-splatting, utils/camera_utils.py camera_to_JSON, main at commit 54c035f): a camera
  is 'position' (its centre in the world) and 'rotation' (3 × 3, camera to world, columns the camera's x, y, z) in
  the OpenCV/COLMAP convention (x right, y down, z forward), which is the REP-103 optical frame; written at the joint
  state q (home by default), so a link-mounted camera's world pose holds for that pose only.

Known limits: as core/tolerance.py; only each sensor's primary imager (the one it is calibrated by) gets optical
frames (the others move rigidly with the body: apply delta_body); fx, fy, cx, cy are the lens model's at the
sensor's stream and say nothing about distortion (lens_model names it).
"""
from __future__ import annotations

import json
import os

import numpy as np

from .. import tolerance as TL
from ..frames import quat_from_matrix
from ..jobs import StudyError
from . import exporter

DEFAULTS = {'n': 100, 'seed': 0, 'scale': 1.0, 'distribution': 'uniform'}
MAX_N = 100_000
FOLDER = 'perturbations'
CONVENTIONS = {
    'd': '[dx, dy, dz] metres along the mount parent frame\'s axes, then [droll, dpitch, dyaw] degrees about the '
         'sensor body\'s own x, y, z axes applied in that order (R\' = R · Rx · Ry · Rz)',
    'delta_body': 'T_body_perturbed = T_body · delta_body (DexVantage core/sim/render extrinsic_delta)',
    'mount': 'the perturbed sensor body pose in its mount parent frame; quaternion (x, y, z, w)',
    'isaac_lab': 'CameraCfg.OffsetCfg in the mount parent frame, convention "ros" (the optical frame: z forward, x '
                 'right, y down); rot_xyzw for Isaac Lab v3.0 and later, rot_wxyz for v2.x',
    'gs': '3DGS camera: position = camera centre in the world, rotation = camera-to-world 3 × 3 (OpenCV axes), at '
          'the joint state the export was given (home by default)',
    'body_frame': 'REP-103: x forward, y left, z up; world = the primary robot base',
}


def _opts(options: dict) -> dict:
    out = dict(DEFAULTS)
    for k in ('n', 'seed', 'scale'):
        if options.get(k) is None:
            continue
        try:
            v = float(options[k])
        except (TypeError, ValueError):
            raise StudyError(f'The perturbation export\'s {k} must be a number; it is {options[k]!r}.') from None
        if not np.isfinite(v) or v < 0:
            raise StudyError(f'The perturbation export\'s {k} must be a positive number; it is {options[k]!r}.')
        out[k] = int(round(v)) if k in ('n', 'seed') else v
    if not 1 <= out['n'] <= MAX_N:
        raise StudyError(f'The perturbation export takes 1 to {MAX_N} draws; it was asked for {out["n"]}.')
    if options.get('distribution') is not None:
        if options['distribution'] not in TL.DISTRIBUTIONS:
            raise StudyError('The perturbation export\'s distribution must be uniform or normal; it is '
                             f'{options["distribution"]!r}.')
        out['distribution'] = options['distribution']
    q = options.get('q')
    out['q'] = {str(k): np.asarray(v, float) for k, v in q.items()} if isinstance(q, dict) else None
    return out


def _r(x, nd=9):
    return [round(float(v), nd) for v in np.asarray(x, float).reshape(-1)]


def _T(T) -> list | None:
    T = np.asarray(T, float)
    return None if not np.isfinite(T).all() else [_r(row) for row in T]


def write(project, out_dir: str, options: dict) -> dict:
    """Write the two files; {'files', 'notes', 'arrays'} (arrays: core.tolerance.perturbed_extrinsics)."""
    o = _opts(options or {})
    A = TL.perturbed_extrinsics(project, o['n'], o['seed'], o['scale'], o['distribution'], q=o['q'])
    folder = os.path.join(out_dir, FOLDER)
    try:
        os.makedirs(folder, exist_ok=True)
    except OSError as e:
        raise StudyError(f'Could not create the folder {FOLDER}: {e.strerror or e}.') from None
    draws = []
    for i in range(A['n']):
        per = []
        for j, s in enumerate(A['sensors']):
            Tpb, Tpo, Two = A['T_parent_body'][i, j], A['T_parent_optical'][i, j], A['T_world_optical'][i, j]
            row = {'d': _r(A['d'][i, j]), 'delta_body': _T(A['delta_body'][i, j]),
                   'mount': {'position': _r(Tpb[:3, 3]), 'quaternion_xyzw': _r(quat_from_matrix(Tpb[:3, :3]))}}
            if np.isfinite(Tpo).all():
                qx = quat_from_matrix(Tpo[:3, :3])
                row['isaac_lab'] = {'pos': _r(Tpo[:3, 3]), 'rot_xyzw': _r(qx), 'rot_wxyz': _r(np.r_[qx[3], qx[:3]]),
                                    'convention': 'ros'}
            if np.isfinite(Two).all():
                row['gs'] = {'position': _r(Two[:3, 3]), 'rotation': [_r(r) for r in Two[:3, :3]]}
            per.append(row)
        draws.append(per)
    doc = {'schema': 'dexvantage.perturbations/1', 'n': A['n'], 'seed': A['seed'], 'scale': A['scale'],
           'distribution': A['distribution'], 'sources': A['sources'], 'conventions': CONVENTIONS,
           'components': A['components'], 'component_units': A['component_units'], 'sensors': A['sensors'],
           'q': None if o['q'] is None else {k: _r(v) for k, v in o['q'].items()},
           'draws': draws}
    jp = os.path.join(folder, 'perturbations.json')
    np_path = os.path.join(folder, 'perturbations.npz')
    try:
        with open(jp, 'w', encoding='utf-8') as fh:
            json.dump(doc, fh, indent=1, ensure_ascii=False)
            fh.write('\n')
        np.savez_compressed(np_path, d=A['d'], delta_body=A['delta_body'], T_parent_body=A['T_parent_body'],
                            T_parent_optical=A['T_parent_optical'], T_world_optical=A['T_world_optical'],
                            sensor_uids=np.array([s['uid'] for s in A['sensors']]),
                            sensor_names=np.array([s['name'] for s in A['sensors']]),
                            seed=np.int64(A['seed']), scale=np.float64(A['scale']),
                            distribution=np.array(A['distribution']))
    except OSError as e:
        raise StudyError(f'Could not write the perturbations: {e.strerror or e}.') from None
    notes = [f'{A["n"]} draws (seed {A["seed"]}, scale {A["scale"]:g}) of {len(A["sensors"])} sensor'
             f'{"" if len(A["sensors"]) == 1 else "s"}: the same seed gives the same draws.']
    for s in A['sources']:
        what = (f'± {s["position_mm"]:g} mm, ± {s["rotation_deg"]:g}° ({s["distribution"]})' if s['source'] == TL.MOUNT
                else f'its calibration covariance ({s.get("session", "")})')
        notes.append(f'{s["name"]}: {what}.')
    if any(s.get('imager') is None for s in A['sensors']):
        notes.append('Sensors without an imager have no optical frames (only the body pose and delta_body).')
    if any(s.get('robot_uid') for s in A['sensors']):
        notes.append('Link-mounted cameras: the Isaac Lab offset is in the link frame (true at every pose); the 3DGS '
                     'world pose is at ' + ('the joint state given.' if o['q'] else 'the robot\'s home pose.'))
    return {'files': [f'{FOLDER}/perturbations.json', f'{FOLDER}/perturbations.npz'], 'notes': notes, 'arrays': A}


@exporter('perturbations', 'Perturbed extrinsics (augmentation and domain randomisation)', needs_robot=False,
          description='n draws of every mount from its tolerance or calibration covariance, as JSON and .npz, for '
                      'Isaac Lab domain randomisation and the 3DGS renderer (spec §8.5).')
def _export(project, out_dir, options, ctx) -> dict:
    res = write(project, out_dir, options)
    return {'files': res['files'], 'notes': res['notes']}

