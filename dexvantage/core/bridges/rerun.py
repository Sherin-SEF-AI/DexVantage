"""The Rerun bridge (spec §9.4): the project's scene in the Rerun viewer, saved as an .rrd file or sent to a live
viewer (started only when the user asks).

What goes out, in the world frame (the primary robot's base, REP-103: x forward, y left, z up, metres; the root
entity 'world' carries ViewCoordinates RIGHT_HAND_Z_UP):

- each robot's links: 'world/robots/<robot>/<link>' with a Transform3D (the link frame in the world) and, below it,
  'mesh' (Mesh3D of the link's visual meshes in the link frame, vertex colours from the model). With a trajectory the
  link transforms are logged at every sample time on the timeline 'time' (seconds from the trajectory's start,
  sampled at rate_hz, default 30 Hz, a labelled default, at most max_frames samples; linear in the joints between
  the trajectory's own samples, as core/robot/trajectory.py); without one, once (static) at q (default home);
- each enabled camera or depth sensor: its body entity 'world/robots/<robot>/<link>/sensors/<sensor>' when it rides
  on a link (so it follows the arm), else 'world/sensors/<sensor>', with a Transform3D (the mount pose); below it one
  entity per imager with the imager's Transform3D in the body, and 'image' with a Pinhole (image_from_camera = K,
  resolution, camera_xyz = RDF: OpenCV's optical axes); K is written in OpenCV's convention (pixel centres at
  integers; Rerun's documentation read on 2026-11-07 does not state a pixel-centre convention, so the frustum may sit
  half a pixel from Rerun's own); 'lens' (a TextDocument naming the lens model and its distortion terms) and, for a
  lens that is not a pinhole, 'field' (LineStrips3D: the true image border unprojected through the lens to frustum_m),
  because Rerun's Pinhole has no distortion;
- each depth sensor whose maker states a min-Z: 'min_z' below its body (Mesh3D and its edges as LineStrips3D: the part
  of the depth imager's field closer than min-Z, core/studies/depth_validity.py blind_zone_local);
- each task volume: 'world/task_volumes/<volume>' (Boxes3D);
- an audit result (core/studies/audit.py), when given: on the timeline 'episode' (the episode index), 'audit/verdict'
  (TextLog: the verdict and the failed flags; level INFO for pass, WARN for fix or recalibrate, ERROR for drop),
  'audit/pass' (Scalars, 1 or 0) and 'audit/flags/<flag>' (Scalars: the cameras failing that requirement in the
  episode, 0 when none; the flag names of core/audit/episode.py FLAG, as the FiftyOne tags).

rerun-sdk 0.42.0 (the pinned optional extra 'rerun'; the installed wheel's source and
https://rerun.io/docs/reference/types/archetypes/pinhole, version 0.42.0, read 2026-11-07):
RecordingStream(application_id, recording_id), .save(path) / .spawn() / .connect_grpc(url), .log(path, archetype,
static=…), .set_time(timeline, duration=… | sequence=…), .reset_time(); archetypes ViewCoordinates,
Transform3D(translation, mat3x3), Mesh3D(vertex_positions, triangle_indices, vertex_colors),
Pinhole(image_from_camera, resolution, camera_xyz, image_plane_distance), LineStrips3D, Boxes3D(centers,
half_sizes), TextDocument, TextLog(text, level), Scalars. Files are read back with
rerun.chunk.RrdReader (read_back below; the validation check formats.rerun_rrd_readback).

Ported from Vantage core/viz_bridge.py (the Rerun part: the frames, the Pinhole with RDF axes, the lens as text, the
true field outline beside the pinhole frustum, the availability test and the three modes). What changed: the
articulated robot over a trajectory instead of a static vehicle body, link-relative sensor entities, min-Z blind zones
and audit flags instead of ground footprints, lidars and the blind volume; a RecordingStream of its own instead of
the global recording (several exports can run in one process).

Known limits: Rerun's Pinhole is a pinhole (a fisheye frustum there is approximate; the true field is the 'field'
lines); meshes are logged once per link (no level of detail), so a large robot makes a large file; contact pads,
task objects, lights, grasp sets and coverage results are not logged; only one trajectory per robot; the audit's
per-frame scores are not logged (per-episode flags only).
"""
from __future__ import annotations

import importlib.util
import math
import os
import re

import numpy as np

from ..jobs import StudyError

PIN = '0.42.0'                      # the version this bridge is written against (pyproject extra 'rerun')
ROOT = 'world'
APP_ID = 'dexvantage'
RATE_HZ = 30.0                      # default trajectory sampling (a labelled default)
MAX_FRAMES = 2000                   # default upper bound on trajectory samples
FRUSTUM_M = 0.15                    # default frustum depth (m) for the Pinhole image plane and the field lines
INSTALL = ('Rerun is not installed: install the optional extra with pip install "dexvantage[rerun]" (rerun-sdk '
           + PIN + '), then try again.')
LEVEL = {'pass': 'INFO', 'fix': 'WARN', 'recalibrate': 'WARN', 'drop': 'ERROR'}


class RerunError(StudyError):
    """The scene cannot be sent to Rerun; the message says why in plain words."""


def available() -> bool:
    """True when the rerun-sdk package can be imported (checked without importing it)."""
    try:
        return importlib.util.find_spec('rerun') is not None
    except (ImportError, ValueError):
        return False


def _rr():
    if not available():
        raise RerunError(INSTALL)
    try:
        import rerun as rr
    except Exception as e:                                # noqa: BLE001 — a broken install: say so in one sentence
        raise RerunError(f'Rerun could not be loaded ({type(e).__name__}: {str(e).splitlines()[0][:160]}). '
                         + INSTALL) from None
    if not (hasattr(rr, 'RecordingStream') and hasattr(rr, 'Transform3D')):
        raise RerunError('The rerun package found is not rerun-sdk. ' + INSTALL)
    return rr


def tag(text: str) -> str:
    t = re.sub(r'[^A-Za-z0-9_-]+', '_', str(text)).strip('_')
    return t[:60] or 'item'


class _Names:
    def __init__(self):
        self.used = set()

    def __call__(self, parent: str, text: str) -> str:
        base = f'{parent}/{tag(text)}'
        p, k = base, 2
        while p in self.used:
            p, k = f'{base}_{k}', k + 1
        self.used.add(p)
        return p


# ====================================================================== the scene (plain data, no rerun)
def _u8(rgba) -> list:
    return [int(round(255 * max(0.0, min(1.0, float(x))))) for x in rgba]


def _robot_meshes(model) -> dict:
    """link -> (V (N, 3) float32 in the link frame, F (M, 3) uint32, colours (N, 4) uint8)."""
    out: dict = {}
    for ln, V, F, c in model.visual_meshes():
        if not len(F):
            continue
        Vs, Fs, Cs = out.get(ln, (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32),
                                  np.zeros((0, 4), np.uint8)))
        F2 = np.asarray(F, np.int64) + len(Vs)
        col = np.tile(np.asarray(_u8(c), np.uint8), (len(V), 1))
        out[ln] = (np.vstack([Vs, np.asarray(V, np.float32)]), np.vstack([Fs, F2.astype(np.uint32)]),
                   np.vstack([Cs, col]))
    return out


def _trajectory_for(project, robot_uid: str, traj_uid: str | None):
    trs = [t for t in project.tasks.trajectories if isinstance(t, dict)]
    if traj_uid:
        t = next((t for t in trs if t.get('uid') == traj_uid), None)
        if t is None:
            raise RerunError('The trajectory named is not in the project.')
        return t if t.get('robot_uid') == robot_uid else None
    return next((t for t in trs if t.get('robot_uid') == robot_uid and t.get('t')), None)


def scene(project, q: dict | None = None, trajectory_uid: str | None = None, audit: dict | None = None,
          rate_hz: float = RATE_HZ, max_frames: int = MAX_FRAMES, frustum_m: float = FRUSTUM_M,
          meshes: bool = True) -> dict:
    """What log() sends, as plain arrays: {'robots': [...], 'sensors': [...], 'blind_zones': [...], 'volumes': [...],
    'audit': [...] or None, 'notes': [...]}. q: {robot uid: q} for robots without a trajectory (None = home)."""
    from .. import library as LB
    from ..robot import trajectory as TJ
    from ..robot.model import RobotModelError, get_model
    from ..studies import depth_validity as DV
    if not (rate_hz > 0 and math.isfinite(rate_hz)):
        raise RerunError('The trajectory sampling rate must be a positive number of samples per second.')
    names = _Names()
    notes: list = []
    robots, link_path = [], {}
    for rob in project.robots:
        try:
            m = get_model(project, rob.uid)
        except RobotModelError as e:
            notes.append(f'{rob.name} is left out: {e}')
            continue
        rpath = names(f'{ROOT}/robots', rob.name)
        tr = _trajectory_for(project, rob.uid, trajectory_uid)
        mesh = _robot_meshes(m) if meshes else {}
        keep = [ln for ln in m.link_names if ln in mesh]
        mounted = {s.mount.parent_frame for v in [project.variant] for s in v.sensors
                   if s.mount.on_link and s.mount.robot_uid == rob.uid}
        frames = mounted | {rob.gripper_frame, *rob.fingertip_frames}    # sensors' links and the gripper's frames
        keep += [ln for ln in m.link_names if ln in frames and ln not in keep]
        paths = {ln: names(rpath, ln) for ln in keep}
        for ln, pth in paths.items():
            link_path[(rob.uid, ln)] = pth
        idx = [m.link_names.index(ln) for ln in keep]
        if tr is not None:
            t0, t1 = TJ.duration(tr)
            n = int(min(max_frames, max(2, math.floor((t1 - t0) * rate_hz) + 1)))
            ts = np.linspace(t0, t1, n) if t1 > t0 else np.array([t0])
            Q = np.array([TJ.on_model(tr, m.joint_names, m.home, float(t)) for t in ts])
            T = m.fk_batch(Q)[:, idx] if len(idx) else np.zeros((len(ts), 0, 4, 4))
            times = ts - t0
            if n == max_frames and (t1 - t0) * rate_hz + 1 > max_frames:
                notes.append(f'{rob.name}: {tr.get("name", "the trajectory")} is sampled at {n} times '
                             f'(the upper bound), {n / max(t1 - t0, 1e-9):.1f} per second.')
        else:
            qq = m.home if not q or q.get(rob.uid) is None else m.clip(np.asarray(q[rob.uid], float))
            T = m.fk_array(qq)[idx][None]
            times = None
        robots.append({'uid': rob.uid, 'name': rob.name, 'path': rpath, 'links': keep, 'paths': paths,
                       'T': T, 'times': times, 'trajectory': (tr or {}).get('name'),
                       'meshes': {ln: mesh[ln] for ln in keep if ln in mesh}})
    sensors, zones = [], []
    for s in project.variant.sensors:
        if not s.enabled or s.kind not in ('camera', 'depth'):
            continue
        ims = LB.sensor_imagers(s, tuple(s.stream) if len(s.stream) == 2 else None)
        if not ims:
            continue
        if s.mount.on_link and (s.mount.robot_uid, s.mount.parent_frame) in link_path:
            parent = link_path[(s.mount.robot_uid, s.mount.parent_frame)]
            T_body = s.mount.pose.matrix()
        elif s.mount.on_link:
            notes.append(f'{s.name} is left out: its robot or link is not in the scene.')
            continue
        else:
            parent = f'{ROOT}/sensors'
            T_body = s.mount.pose.matrix()          # a static mount: its pose is read in the world frame
        spath = names(parent if parent == f'{ROOT}/sensors' else f'{parent}/sensors', s.name)
        imagers = []
        for im in ims:
            cam = im.camera
            ipath = names(spath, im.name)
            K = np.array([[cam.fx, 0.0, cam.cx], [0.0, cam.fy, cam.cy], [0.0, 0.0, 1.0]])
            field = None
            if cam.model != 'pinhole':
                from ..camrender import image_border
                rays = image_border(cam, 24)
                ok = np.isfinite(rays).all(axis=1)
                field = rays[ok] * frustum_m
            imagers.append({'name': im.name, 'role': im.role, 'path': ipath, 'K': K, 'width': cam.width,
                            'height': cam.height, 'model': cam.model, 'dist': list(cam.dist),
                            'T': im.pose_in_body.matrix(), 'field': field})
        sensors.append({'uid': s.uid, 'name': s.name, 'path': spath, 'T': T_body, 'imagers': imagers})
        if s.kind == 'depth' or LB.depth_spec(s) is not None:
            try:
                ds = DV.setup(project, s)
                bz = DV.blind_zone_local(ds)
            except StudyError as e:
                notes.append(f'{s.name}: no min-Z zone ({e}).')
                bz = None
            if bz is not None:
                tris, segs = bz
                zones.append({'sensor': s.name, 'path': f'{spath}/min_z', 'tris': tris, 'segs': segs,
                              'min_z_m': ds.min_z_m, 'note': ds.min_z_note})
    vols = []
    vnames = _Names()
    for v in project.workspace.volumes:
        vols.append({'name': v.name, 'path': vnames(f'{ROOT}/task_volumes', v.name),
                     'center': np.asarray(v.center, float), 'half': np.asarray(v.size, float) / 2})
    aud = None
    if audit:
        from ..audit import episode as AE
        flags = sorted(set(AE.FLAG.values()))
        aud = []
        for e in audit.get('episodes') or []:
            fails = [r for r in e.get('requirements') or [] if not r.get('pass')]
            counts = {f: 0 for f in flags}
            for r in fails:
                f = AE.FLAG.get(r.get('id'))
                if f:
                    counts[f] += 1
            aud.append({'index': int(e.get('index', len(aud))), 'id': str(e.get('id', '')),
                        'verdict': str(e.get('verdict', '')), 'counts': counts,
                        'failed': sorted({AE.FLAG.get(r.get('id'), str(r.get('id'))) for r in fails})})
        if not aud:
            notes.append('The audit result has no episodes to log.')
    return {'robots': robots, 'sensors': sensors, 'blind_zones': zones, 'volumes': vols, 'audit': aud,
            'notes': notes, 'frustum_m': float(frustum_m)}


# ====================================================================== logging
def log(rec, rr, sc: dict) -> list:
    """Log a scene() dict to a RecordingStream rec (rr: the rerun module); returns the entity paths logged."""
    paths: list = []

    def put(path, arch, static=True):
        rec.log(path, arch, static=static)
        if path not in paths:
            paths.append(path)

    def tf(T):
        T = np.asarray(T, float)
        return rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3])

    put(ROOT, rr.ViewCoordinates.RIGHT_HAND_Z_UP)
    for r in sc['robots']:
        for ln, (V, F, C) in r['meshes'].items():
            put(f'{r["paths"][ln]}/mesh', rr.Mesh3D(vertex_positions=V, triangle_indices=F, vertex_colors=C))
        if r['times'] is None:
            for k, ln in enumerate(r['links']):
                put(r['paths'][ln], tf(r['T'][0, k]))
    for s in sc['sensors']:
        put(s['path'], tf(s['T']))
        for im in s['imagers']:
            put(im['path'], tf(im['T']))
            put(f'{im["path"]}/image', rr.Pinhole(image_from_camera=im['K'], resolution=[im['width'], im['height']],
                                                   camera_xyz=rr.ViewCoordinates.RDF,
                                                   image_plane_distance=sc['frustum_m']))
            put(f'{im["path"]}/lens', rr.TextDocument(
                f'{im["model"]} lens, {im["width"]} × {im["height"]} px, fx {im["K"][0, 0]:.3f}, fy '
                f'{im["K"][1, 1]:.3f}, cx {im["K"][0, 2]:.3f}, cy {im["K"][1, 2]:.3f} px (OpenCV: pixel centres at '
                f'integers); distortion {[round(x, 8) for x in im["dist"]] or "none"}'
                + ('; the Pinhole frustum leaves the distortion out: see field' if im['field'] is not None else '')))
            if im['field'] is not None and len(im['field']) > 1:
                put(f'{im["path"]}/field', rr.LineStrips3D([np.vstack([im['field'], im['field'][:1]])]))
    for z in sc['blind_zones']:
        tri = np.asarray(z['tris'], float).reshape(-1, 3)
        put(z['path'], rr.Mesh3D(vertex_positions=tri, triangle_indices=np.arange(len(tri)).reshape(-1, 3),
                                 albedo_factor=[209, 59, 59, 90]))
        put(f'{z["path"]}/edges', rr.LineStrips3D(list(np.asarray(z['segs'], float)), colors=[[209, 59, 59, 255]]))
    for v in sc['volumes']:
        put(v['path'], rr.Boxes3D(centers=[v['center']], half_sizes=[v['half']], labels=[v['name']],
                                  colors=[[57, 135, 229, 255]]))
    for r in sc['robots']:
        if r['times'] is None:
            continue
        for i, t in enumerate(r['times']):
            rec.set_time('time', duration=float(t))
            for k, ln in enumerate(r['links']):
                put(r['paths'][ln], tf(r['T'][i, k]), static=False)
        rec.reset_time()
    if sc.get('audit'):
        for e in sc['audit']:
            rec.set_time('episode', sequence=int(e['index']))
            name = str(e['id'] or e['index'])
            text = (name if name.lower().startswith('episode') else f'episode {name}') + \
                f': {e["verdict"] or "not audited"}'
            if e['failed']:
                text += ' (' + ', '.join(e['failed']) + ')'
            put('audit/verdict', rr.TextLog(text, level=LEVEL.get(e['verdict'], 'WARN')), static=False)
            put('audit/pass', rr.Scalars(1.0 if e['verdict'] == 'pass' else 0.0), static=False)
            for f, n in e['counts'].items():
                put(f'audit/flags/{f}', rr.Scalars(float(n)), static=False)
        rec.reset_time()
    return paths


def stream(project, mode: str = 'save', path: str | None = None, url: str | None = None, audit: dict | None = None,
           **kw) -> dict:
    """Send the scene to Rerun: mode 'save' (an .rrd file at path), 'spawn' (start a viewer: only on the user's
    action) or 'connect' (a running viewer at url, else Rerun's default). Returns {'text', 'paths', 'notes', 'path'}.
    Raises RerunError (with the install command when rerun-sdk is missing)."""
    if mode not in ('save', 'spawn', 'connect'):
        raise RerunError(f'Unknown Rerun mode {mode!r}: save, spawn or connect.')
    if mode == 'save':
        if not path:
            raise RerunError('Saving to Rerun needs a file name ending in .rrd.')
        folder = os.path.dirname(os.path.abspath(path))
        if not os.path.isdir(folder):
            raise RerunError(f'The folder for {os.path.basename(path)} does not exist: choose one that does.')
    rr = _rr()
    sc = scene(project, audit=audit, **kw)
    rec = rr.RecordingStream(APP_ID)
    try:
        if mode == 'save':
            rec.save(path)
        elif mode == 'spawn':
            rec.spawn()
        else:
            rec.connect_grpc(url) if url else rec.connect_grpc()
    except Exception as e:                                # noqa: BLE001 — the sink: one plain sentence
        raise RerunError(f'Rerun could not {"write " + os.path.basename(path) if mode == "save" else "reach a viewer"}'
                         f' ({str(e).splitlines()[0][:160] if str(e) else type(e).__name__}).') from None
    try:
        paths = log(rec, rr, sc)
        rec.flush()
    finally:
        try:
            rec.disconnect()
        except Exception:                                 # noqa: BLE001
            pass
    where = {'save': os.path.basename(path or ''), 'spawn': 'a new Rerun viewer',
             'connect': url or 'the Rerun viewer'}[mode]
    n_links = sum(len(r['links']) for r in sc['robots'])
    n_cams = sum(len(s['imagers']) for s in sc['sensors'])
    moving = [r for r in sc['robots'] if r['times'] is not None]
    text = (f'Sent {len(sc["robots"])} robot{"" if len(sc["robots"]) == 1 else "s"} ({n_links} links'
            + (f', {sum(len(r["times"]) for r in moving)} poses over {", ".join(r["trajectory"] for r in moving)}'
               if moving else '') + f'), {n_cams} camera frustum{"" if n_cams == 1 else "s"}, '
            f'{len(sc["blind_zones"])} min-Z zone{"" if len(sc["blind_zones"]) == 1 else "s"}, '
            f'{len(sc["volumes"])} task volume{"" if len(sc["volumes"]) == 1 else "s"}'
            + (f' and {len(sc["audit"])} audited episode{"" if len(sc["audit"]) == 1 else "s"}'
               if sc.get('audit') else '') + f' to {where}.')
    return {'text': text, 'paths': paths, 'notes': sc['notes'], 'path': path if mode == 'save' else None}


def batch(project, out: str, audit_path: str | None = None, audit_doc: dict | None = None) -> str:
    """--rerun OUT.rrd [--rerun-audit AUDIT.json]: save the scene (with the audit of this run, or the one in
    audit_path); the line to print. Raises RerunError in a plain sentence."""
    import json
    if not str(out).lower().endswith('.rrd'):
        raise RerunError(f'--rerun writes an .rrd file: give a name ending in .rrd, not {os.path.basename(out)}.')
    if audit_path:
        try:
            with open(audit_path, encoding='utf-8') as fh:
                audit_doc = json.load(fh)
        except (OSError, ValueError) as e:
            raise RerunError(f'The audit result {os.path.basename(audit_path)} could not be read '
                             f'({getattr(e, "strerror", None) or e}).') from None
        if not isinstance(audit_doc, dict) or audit_doc.get('study') != 'audit':
            raise RerunError(f'{os.path.basename(audit_path)} is not an audit result (write one with --audit).')
    res = stream(project, 'save', out, audit=audit_doc)
    return res['text'].replace(f'to {os.path.basename(out)}.', f'→ {out}') + \
        ''.join(f'\n  · {n}' for n in res['notes'])


# ====================================================================== reading back
def read_back(path: str) -> dict:
    """An .rrd file read with rerun's own reader (rerun.chunk.RrdReader): {entity path: {'static': {column: value},
    'rows': [{column: value, timeline: value}]}} with the archetype columns ('Transform3D:translation', …) as plain
    lists. For the validation and the tests."""
    _rr()
    from rerun.chunk import RrdReader
    out: dict = {}
    for c in RrdReader(path).stream().to_chunks():
        ent = out.setdefault(c.entity_path, {'static': {}, 'rows': []})
        b = c.to_record_batch()
        cols = [n for n in b.schema.names if n != 'rerun.controls.RowId']
        data = {n: _plain(b.column(n)) for n in cols}
        nrows = b.num_rows
        if c.is_static:
            for n in cols:
                if data[n] and data[n][-1] is not None:
                    ent['static'][n] = data[n][-1]
        else:
            for i in range(nrows):
                ent['rows'].append({n: data[n][i] for n in cols})
    return out


def _plain(col) -> list:
    """A pyarrow column as Python values; times and durations as integer nanoseconds (or sequence numbers)."""
    import pyarrow as pa
    if pa.types.is_temporal(col.type):
        col = col.cast(pa.int64())
    return col.to_pylist()


def first(value):
    """The first instance of a component column value as read_back gives it (a list of instances)."""
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    return value
