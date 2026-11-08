"""Exporters that wrap what other modules already write (spec §9.2, §9.4): Plücker ray maps (core/plucker.py),
bracket STL (core/brackets.py, STEP through gmsh when it is installed), the FiftyOne export of a dataset audit
(core/audit/report.py) and a synthetic LeRobot dataset (core/sim/episodes.py).

    'plucker'           options: policy (uid; default the first policy with a camera in use), frame ('base'),
                        robot_uid, dtype ('float32'), order ('dm'), formats (['npy', 'safetensors']), q
    'brackets'          options: step (False), step_res_mm (None: the STL's mesh), step_max_triangles
                        (STEP_MAX_TRIANGLES, a default)
    'fiftyone'          options: dataset (a path or hf://datasets/<org>/<name>; default the dataset the project's
                        audit uses), media ('clips')
    'lerobot_dataset'   options: the synthetic-episode settings (core/sim/episodes.settings_of)

STEP. A printed bracket is an implicit solid meshed into triangles (core/meshing.py), so it has no parametric
surfaces to hand a CAD tool. With gmsh installed (pip install "dexvantage[step]"), the STL's triangles become the
planar faces of one OpenCASCADE solid (each unique vertex a point, each edge a line shared by its two faces, one
surface loop), which gmsh writes as STEP: a faceted solid with the mesh's exact geometry, not a smooth CAD model.
Without gmsh the STEP request is refused in one sentence and nothing else is written.

Known limits: each wrapper has the limits of the module it wraps; a faceted STEP is large (a face per triangle) and
is refused above step_max_triangles triangles; the FiftyOne export runs the audit study on the dataset first (it is
not cached between exports).
"""
from __future__ import annotations

import os

from ..jobs import StudyError
from . import _common as C
from . import exporter

STEP_MAX_TRIANGLES = 20000        # a default: larger faceted STEP files are refused (a face per triangle)


# ---------------------------------------------------------------------------------------------------------------
@exporter('plucker', 'Plücker ray maps', needs_robot=False,
          description='Per camera of a policy, the per-pixel Plücker rays (o × d, d) at the policy\'s input '
                      'resolution, as .npy and .safetensors with a JSON beside each (core/plucker.py).')
def export_plucker_maps(project, out_dir, options, ctx):
    from .. import plucker as PL
    from .. import policies as PO
    pols = list(getattr(project, 'policies', []) or [])
    if not pols:
        raise StudyError('Plücker ray maps need a policy: add one on the Policies page (its input size sets the '
                         'map\'s resolution).')
    uid = options.get('policy') or ''
    if uid:
        pol = PO.policy(project, uid)
        if pol is None:
            raise StudyError('The policy chosen for the Plücker export is not in the project any more.')
    else:
        pol = next((p for p in pols if any(s.enabled for s in PO.policy_cameras(project, p))), pols[0])
    q, q_text = C.joint_state(project, options)
    folder = os.path.join(out_dir, 'plucker')
    paths = PL.export_plucker(project, folder, pol.uid, q=q, frame=str(options.get('frame') or 'base'),
                              robot_uid=options.get('robot_uid') or None, dtype=str(options.get('dtype') or 'float32'),
                              order=str(options.get('order') or 'dm'),
                              formats=tuple(options.get('formats') or ('npy', 'safetensors')))
    notes = [f'Ray maps of {pol.name} at {q_text}: a camera on a link moves with the arm, so export at the pose the '
             'policy sees.']
    return {'files': [os.path.relpath(p, out_dir).replace(os.sep, '/') for p in paths], 'notes': notes}


# ---------------------------------------------------------------------------------------------------------------
class _quiet_stdout:
    """Send file descriptor 1 to the null device for the block (a C library's console output)."""

    def __enter__(self):
        import sys
        try:
            sys.stdout.flush()
            self.saved = os.dup(1)
            null = os.open(os.devnull, os.O_WRONLY)
            os.dup2(null, 1)
            os.close(null)
        except OSError:
            self.saved = None
        return self

    def __exit__(self, *exc):
        if self.saved is not None:
            os.dup2(self.saved, 1)
            os.close(self.saved)
        return False


def step_from_mesh(V, F, path: str):
    """Write a triangle mesh (V in mm, F) as a faceted STEP solid through gmsh's OpenCASCADE kernel (module
    docstring). Raises StudyError when gmsh is not installed."""
    import numpy as np
    try:
        import gmsh
    except ImportError:
        from ...extras import missing
        raise StudyError(missing('step', 'STEP export needs the optional gmsh package, which is not installed: pip '
                                 'install "dexvantage[step]", or export STL only.', ', or export STL only')) from None
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    started = gmsh.isInitialized() if hasattr(gmsh, 'isInitialized') else False
    if not started:
        gmsh.initialize([], readConfigFiles=False)
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.model.add('dexvantage_bracket')
        occ = gmsh.model.occ
        pts = [occ.addPoint(*map(float, v)) for v in V]
        edges = {}
        faces = []
        for tri in F:
            loop = []
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                key = (min(a, b), max(a, b))
                if key not in edges:
                    edges[key] = occ.addLine(pts[key[0]], pts[key[1]])
                loop.append(edges[key] if a < b else -edges[key])
            faces.append(occ.addPlaneSurface([occ.addCurveLoop(loop)]))
        sl = occ.addSurfaceLoop(faces, sewing=True)
        occ.addVolume([sl])
        occ.synchronize()
        with _quiet_stdout():                    # OpenCASCADE's STEP writer prints its banner on the C stdout
            gmsh.write(path)
    finally:
        gmsh.model.remove()
        if not started:
            gmsh.finalize()


@exporter('brackets', 'Bracket STL (and STEP)', needs_robot=False,
          description='Each sensor\'s printable bracket as a watertight STL with its JSON sidecar (dimensions, print '
                      'notes, provenance); STEP as a faceted solid when gmsh is installed.')
def export_brackets(project, out_dir, options, ctx):
    from .. import brackets as B
    sensors = [s for s in project.variant.sensors if s.enabled and (s.mount.bracket or '').strip()
               and s.mount.bracket != 'none']
    if not sensors:
        raise StudyError('No sensor has a bracket type: choose one per sensor on Analysis ▸ Brackets (Apply sets the '
                         'suggested ones).')
    want_step = bool(options.get('step', False))
    if want_step:
        try:
            import gmsh  # noqa: F401
        except ImportError:
            from ...extras import missing
            raise StudyError(missing('step', 'STEP export needs the optional gmsh package, which is not installed: '
                                     'pip install "dexvantage[step]", or export STL only (step off).',
                                     ', or export STL only (step off)')) from None
    cap = int(options.get('step_max_triangles') or STEP_MAX_TRIANGLES)
    files, notes, used = [], [], set()
    st = (project.analysis.studies.get('brackets') or {}).get('sensors') or {}
    for k, s in enumerate(sensors):
        ctx.report(k / len(sensors), f'Bracket of {s.name}')
        if ctx.cancelled():
            raise StudyError('The bracket export was cancelled.')
        spec = B.spec_for(project, s)
        extra = st.get(s.uid) or {}
        spec = B.BracketSpec.from_dict(dict(extra, kind=spec.kind, pattern=extra.get('pattern') or spec.pattern))
        try:
            row = B.build(project, s, spec, clear=False)
        except B.BracketError as e:
            notes.append(f'{s.name}: no bracket ({e}).')
            continue
        if row.get('_mesh') is None:
            notes.append(f'{s.name}: no bracket ({row.get("message") or "nothing to mesh"}).')
            continue
        stem = B.safe_name(s.name)
        n = 2
        while stem in used:
            stem, n = f'{B.safe_name(s.name)}_{n}', n + 1
        used.add(stem)
        os.makedirs(os.path.join(out_dir, 'brackets'), exist_ok=True)
        try:
            paths = B.export_stl(row, project, os.path.join(out_dir, 'brackets', stem + '.stl'))
        except OSError as e:
            raise StudyError(f'Could not write the bracket of {s.name}: {e.strerror or e}.') from None
        files += [os.path.relpath(p, out_dir).replace(os.sep, '/') for p in paths]
        if row.get('flags'):
            notes.append(f'{s.name}: {"; ".join(str(f) for f in row["flags"])}.')
        if want_step:
            V, F = row['_mesh']
            res = options.get('step_res_mm')
            if res:
                try:
                    srow = B.build(project, s, B.BracketSpec.from_dict(dict(spec.to_dict(), res_mm=float(res))),
                                   clear=False)
                except (B.BracketError, ValueError, TypeError) as e:
                    raise StudyError(f'The STEP resolution {res!r} mm cannot be used for {s.name}: {e}.') from None
                if srow.get('_mesh') is not None:
                    V, F = srow['_mesh']
                notes.append(f'{s.name}: the STEP is meshed at {float(res):g} mm (the STL at {spec.res_mm:g} mm), so '
                             'its edges are cut by up to that grid step.')
            if len(F) > cap:
                notes.append(f'{s.name}: no STEP, its mesh has {len(F):,} triangles (more than {cap:,}, a default: a '
                             'faceted STEP has a face per triangle, about 2.3 kB and 0.45 ms each). Set step_res_mm '
                             '(e.g. 1.5) for a coarser STEP, or raise step_max_triangles.')
                continue
            rel = f'brackets/{stem}.step'
            step_from_mesh(V, F, os.path.join(out_dir, rel))
            files.append(rel)
    if want_step and any(f.endswith('.step') for f in files):
        notes.append('STEP files are faceted solids (each STL triangle a planar face, millimetres, the bracket frame), '
                     'not parametric CAD models.')
    if not files:
        raise StudyError('No bracket could be generated: ' + ' '.join(notes))
    return {'files': files, 'notes': notes}


# ---------------------------------------------------------------------------------------------------------------
@exporter('fiftyone', 'FiftyOne dataset of an audit', needs_robot=False,
          description='Audits a recorded dataset and writes its per-episode scores (blur, desync, gripper-hidden '
                      'frames, mount slip) as FiftyOne sample tags and fields (core/audit/report.py).')
def export_fiftyone(project, out_dir, options, ctx):
    from ..audit import report as RP
    from ..studies.audit import SID, dataset_path, use_dataset
    path = str(options.get('dataset') or '').strip() or str(dataset_path(project) or '').strip()
    if not path:
        raise StudyError('The FiftyOne export needs the recorded dataset to audit: give its folder (option dataset, '
                         'or --dataset PATH in batch), or add the dataset to the project on the Dataset audit page.')
    snap = project.snapshot()
    use_dataset(snap, path, SID)
    st = dict(snap.analysis.studies.get(SID) or {})
    st['per_frame'] = True
    snap.analysis.studies[SID] = st
    from .. import jobs
    ctx.report(0.05, 'Auditing the dataset')
    doc = jobs.run_study(SID, snap)
    folder = os.path.join(out_dir, 'fiftyone')
    res = RP.export_fiftyone(doc, path, folder, snap, media=str(options.get('media') or 'clips'))
    files = []
    for root, _d, fs in os.walk(folder):
        for f in sorted(fs):
            files.append(os.path.relpath(os.path.join(root, f), out_dir).replace(os.sep, '/'))
    return {'files': sorted(files), 'unstamped': [f for f in files if f.endswith('.json')],
            'notes': [f'{res.get("samples", "?")} samples from the audit of '
                      f'{os.path.basename(os.path.normpath(path))}; its JSON files are FiftyOne\'s and are left as '
                      'written (the manifest carries the provenance).']}


# ---------------------------------------------------------------------------------------------------------------
@exporter('lerobot_dataset', 'Synthetic LeRobot dataset', needs_robot=True,
          description='Synthetic LeRobot v2.1 or v3.0 episodes rendered in MuJoCo with known faults and their ground '
                      'truth (core/sim/episodes.py), in lerobot_dataset/.')
def export_lerobot_dataset(project, out_dir, options, ctx):
    from ..sim import episodes as SE
    folder = os.path.join(out_dir, 'lerobot_dataset')
    if os.path.exists(folder) and os.listdir(folder):
        raise StudyError('The folder lerobot_dataset in the export folder is not empty: choose a new export folder.')
    s = SE.generate(project, folder, dict(options), progress=lambda f, t='': ctx.report(f, t), cancel=ctx.cancelled)
    files = []
    for root, _d, fs in os.walk(folder):
        for f in sorted(fs):
            files.append(os.path.relpath(os.path.join(root, f), out_dir).replace(os.sep, '/'))
    return {'files': sorted(files), 'unstamped': [f for f in files if f.endswith('.json') and
                                                  not f.endswith('dexvantage_truth.json')],
            'notes': [SE.summary_text({k: v for k, v in s.items() if k != 'truth'}),
                      'The dataset\'s JSON files are left as LeRobot defines them (the manifest carries the '
                      'provenance; dexvantage_truth.json, DexVantage\'s own, is stamped).']}
