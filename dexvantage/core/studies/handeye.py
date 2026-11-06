"""Hand-eye calibration studies (spec §8.1): the pose plan and the solve of a recorded session, with the coach.

handeye_plan (batch --handeye-plan OUT.json): core/calib/planner.py on the project: the camera (settings 'sensor',
default the first camera on a robot link, else the first static one), the board (settings 'board', a board uid of
project.calibration; default the first, else DexVantage's default placement), the robot's current joint state
(settings 'q', passed by the page; the plan's tour starts there) and every PlanSettings field (n_poses, pixel_sigma,
need_mm, need_deg, ...). Output: the planner's document (planned poses with joint vectors, predicted σ of X and Z in
mm and degrees, the 12 × 12 covariance, the status against the requirement, what one more pose gains).

handeye_solve (batch --handeye-solve OUT.json): core/calib/handeye.py's bundle adjustment on a capture session of
project.calibration (settings 'session', default the newest session with detections), or, for batch runs, on a folder
captured at run time (settings 'folder' and 'joints', with 'tolerance_s' and 'unit'; core/calib/capture.py). Solver
settings: 'joint_offsets' (estimate the robot's joint zero offsets), 'intrinsics' (refine the camera's lens model),
'pixel_sigma', 'outlier_k', 'outlier_floor_px'. Output: the solve (HandEyeResult.to_dict), per view its image, corners,
RMS and whether it was used, the status of the camera's σ against the requirement, the mount pose Apply writes (the
sensor body in its parent frame, and its change from the current mount), and the coach's next pose (core/calib/
coach.py, settings 'coach', default on).

Several sessions together (settings 'sessions': a list of session uids, or 'all' for every session with detections;
two or more): core/calib/capture.py sessions_problem merges them into one bundle adjustment (a camera or board shared
by sessions is estimated once; images taken at the same joint vector are one robot pose), with the same solver
settings ('joint_offsets' then estimates the offsets of every robot in it, 'intrinsics' refines every camera). The
output adds 'cameras' (per camera: its σ, status against the requirement and the mount Apply writes) and 'boards';
'camera' and 'apply' are the camera furthest from the requirement, 'applies' every camera's mount (the page's Apply
writes them all as one undo step); the views name their session. A camera that no view links to the others (a static
camera that sees only a board no robot-mounted camera sees) is refused by name. The coach advises the session of the
robot-mounted camera (or held board) furthest from the requirement; a static camera seeing a static board ('static')
is located through the board, and the notes say the coach cannot help it.

Known limits: those of core/calib/planner.py, coach.py, capture.py and handeye.py (docs/models/handeye.md): one camera
and one board per plan and per captured session (several sessions solve together), joint readings taken at the
image's instant, the covariance a lower bound when detected corners are correlated.
"""
from __future__ import annotations

import math

import numpy as np

from .. import jobs
from ..jobs import StudyError

PLAN_ID = 'handeye_plan'
SOLVE_ID = 'handeye_solve'


def plan_settings(project) -> dict:
    return dict(project.analysis.studies.get(PLAN_ID) or {})


def solve_settings(project) -> dict:
    return dict(project.analysis.studies.get(SOLVE_ID) or {})


@jobs.study(PLAN_ID, 'Hand-eye pose plan', quick=False, batch='handeye-plan', needs_robot=True)
def plan_study(project, ctx: jobs.Ctx) -> dict:
    """The hand-eye pose plan (module docstring)."""
    from ..calib import planner as PL
    st = plan_settings(project)
    q = st.get('q') if isinstance(st.get('q'), dict) else None
    rig = PL.rig_from_project(project, st.get('sensor') or None, st.get('board') or None, q=q)
    ps = PL.PlanSettings.from_dict(st)
    try:
        pl = PL.plan(rig, ps, start=rig.q_hold, progress=ctx.report, cancel=ctx.cancelled)
    except ValueError as e:
        raise StudyError(str(e)) from None
    doc = dict(pl.doc)
    doc['sensor_uid'] = getattr(rig.sensor, 'uid', '')
    doc['q'] = {k: [float(x) for x in np.asarray(v, float).reshape(-1)] for k, v in (q or {}).items()}
    return doc


def _views_table(res_doc: dict, session: dict, index: list) -> list:
    poses = session.get('poses') or []
    rows = []
    for v in res_doc.get('views') or []:
        k = index[v['pose']] if 0 <= v.get('pose', -1) < len(index) else None
        p = poses[k] if k is not None and k < len(poses) else {}
        rows.append({'name': v.get('name'), 'image': p.get('image', ''), 'session_pose': k, 'corners': v.get('n'),
                     'used': bool(v.get('used')), 'rms_px': v.get('rms_px'), 'note': v.get('note', '') or
                     p.get('note', ''), 'dt_ms': None if p.get('dt_s') is None else 1000 * float(p['dt_s'])})
    seen = {r['session_pose'] for r in rows}
    for k, p in enumerate(poses):
        if k not in seen:
            rows.append({'name': p.get('name') or f'pose {k + 1}', 'image': p.get('image', ''), 'session_pose': k,
                         'corners': len(p.get('ids') or []), 'used': False, 'rms_px': None,
                         'note': p.get('note') or ('no detection' if p.get('q') else 'no joint reading matched'),
                         'dt_ms': None if p.get('dt_s') is None else 1000 * float(p['dt_s'])})
    return rows


@jobs.study(SOLVE_ID, 'Hand-eye solve', quick=False, batch='handeye-solve', needs_robot=True)
def solve_study(project, ctx: jobs.Ctx) -> dict:
    """The hand-eye solve of a session, and the coach (module docstring)."""
    from ..calib import capture as CAP
    from ..calib import handeye as HE
    from ..calib import planner as PL
    from ..calib import records as RC
    from ..calib.axxb import transform_error
    st = solve_settings(project)
    notes = []
    session = None
    if st.get('sessions') and not (st.get('folder') and st.get('joints')):
        return _joint_solve(project, ctx, st)
    if st.get('folder') and st.get('joints'):
        sensor = project.variant.sensor(st.get('sensor')) if st.get('sensor') else None
        if sensor is None:
            sensor = next((s for s in project.variant.sensors if s.kind in ('camera', 'depth') and s.mount.on_link),
                          None) or next((s for s in project.variant.sensors if s.kind in ('camera', 'depth')), None)
        if sensor is None:
            raise StudyError('The hand-eye solve needs a camera in the rig.')
        try:
            CAP.check_source(str(st.get('source') or 'folder'))
            session = CAP.capture_folder(project, sensor, str(st['folder']), str(st['joints']),
                                         RC.find_board(project, st.get('board')),
                                         float(st.get('tolerance_s') or CAP.TOLERANCE_S), str(st.get('unit') or 'rad'),
                                         progress=lambda f, t='': ctx.report(0.3 * f, t), cancel=ctx.cancelled)
        except CAP.CaptureError as e:
            raise StudyError(str(e)) from None
        notes.append('Captured at run time from the folder in the settings (not stored in the project).')
    else:
        session = RC.find_session(project, st.get('session') or None, detected=not st.get('session'))
        if session is None:
            raise StudyError('There is no capture session to solve: import a folder of board images and a '
                             'joint_states file on the Hand-eye solve page first.' if not st.get('session') else
                             'The session chosen is not in the project any more.')
    if session.get('source', {}).get('kind') == 'plan' and not any(p.get('ids') for p in session.get('poses') or []):
        raise StudyError('This session is a plan: it has joint vectors but no images. Record the poses and import '
                         'the images and joint states.')
    try:
        prob, info = CAP.session_problem(project, session)
    except CAP.CaptureError as e:
        raise StudyError(str(e)) from None
    cname, bname = info['camera'], info['board']
    if st.get('intrinsics'):
        prob.cameras[0].intrinsics = True
    hs = HE.Settings(joint_offsets=(info['robot_uid'],) if st.get('joint_offsets') else (),
                     pixel_sigma=float(st['pixel_sigma']) if st.get('pixel_sigma') else None,
                     outlier_k=float(st.get('outlier_k') or HE.OUTLIER_K),
                     outlier_floor_px=float(st.get('outlier_floor_px') or HE.OUTLIER_FLOOR_PX))
    res = HE.solve(prob, hs, progress=lambda f, t='': ctx.report(0.3 + 0.5 * f, t))
    rdoc = res.to_dict()
    ps = PL.PlanSettings.from_dict(st)
    doc = {'session': {'uid': session.get('uid'), 'name': session.get('name'), 'mode': info['mode'],
                       'poses': len(session.get('poses') or []), 'source': session.get('source', {}),
                       'stored': not (st.get('folder') and st.get('joints'))},
           'ok': res.ok, 'message': res.message, 'result': rdoc, 'notes': notes,
           'requirement': {'need_mm': ps.need_mm, 'need_deg': ps.need_deg,
                           'default': {'need_mm', 'need_deg'} <= set(ps.to_dict()['defaults'])},
           'views': _views_table(rdoc, session, info['pose_index']) if res.ok else []}
    sensor = info['sensor']
    if not res.ok:
        doc.update(status='critical', word='not solved', advice=res.message, apply=None, coach=None)
        return doc
    sx = PL.sigma_dict(res.cov_X[cname])
    ratio, key = PL.worst(sx, ps.need_mm, ps.need_deg)
    sev, word = PL.verdict(ratio)
    X = res.X[cname]
    dt, dr = transform_error(X, info['X0'])
    imager = RC.sensor_imager(sensor)
    body = RC.mount_pose_from_X(sensor, imager, X)
    parent = sensor.mount.parent_frame
    robot_uid = sensor.mount.robot_uid
    if info['mode'] == 'eye_in_hand':
        parent, robot_uid = info['link'], info['robot_uid']
    elif sensor.mount.on_link:
        parent, robot_uid = 'world', ''
    cal = None
    if cname in res.intrinsics:
        cal = {'imager': imager.name, 'record': RC.camera_calibration_from(
            res.intrinsics[cname]['model'], f'hand-eye solve of session "{session.get("name")}"')}
    rec = info['board_record']
    doc.update(
        status=sev, word=word,
        worst={'ratio': PL._r(ratio), 'key': key, 'value': PL._r(sx.get(key, math.inf), 5),
               'axis': PL.AXIS_WORDS.get(key.split('_')[0], key) if key else ''},
        camera={'name': cname, 'sensor_uid': sensor.uid, 'imager': imager.name, 'sigma': PL._sig_doc(sx),
                'frame': res.frames.get(cname, '')},
        board={'name': bname, 'uid': rec.get('uid', ''), 'sigma': PL._sig_doc(PL.sigma_dict(res.cov_Z[bname])),
               'frame': res.frames.get(bname, '')},
        apply={'sensor_uid': sensor.uid, 'parent_frame': parent, 'robot_uid': robot_uid, 'pose': body.to_dict(),
               'change_mm': 1000 * dt, 'change_deg': math.degrees(dr), 'calibration': cal,
               'board_uid': rec.get('uid', '') if RC.find_board(project, rec.get('uid')) else '',
               'board_pose': {'parent_frame': ((rec.get('mount') or {}).get('parent_frame') or 'world'),
                              'robot_uid': (rec.get('mount') or {}).get('robot_uid') or '',
                              'pose': PL._pose(res.Z[bname])}})
    used = sum(1 for v in doc['views'] if v['used'])
    doc['advice'] = (f'{used} views used, RMS {res.rms_px:.3f} px. The camera moves {1000 * dt:.2f} mm and '
                     f'{math.degrees(dr):.3f}° from its current mount; Apply writes the calibrated mount.')
    sim = (session.get('source') or {}).get('simulated')
    if isinstance(sim, dict):          # core/calib/simcapture.py: say so, and how far the solve is from the truth
        text = (f'This session is simulated (a test of the pipeline, not a measurement): the camera was moved '
                f'{float(sim.get("offset_norm_mm") or 0):.2f} mm and {float(sim.get("offset_angle_deg") or 0):.2f}° '
                'from its mount.')
        if sim.get('X_true') is not None:
            et, er = transform_error(X, np.asarray(sim['X_true'], float))
            doc['simulated'] = {'error_mm': 1000 * et, 'error_deg': math.degrees(er)}
            text += f' The solve is {1000 * et:.3f} mm and {math.degrees(er):.4f}° from that truth.'
        doc['notes'].insert(0, text)
        doc['advice'] += ' Simulated capture: a test, not a measurement.'
    doc['coach'] = None
    if st.get('coach', True):
        ctx.report(0.85, 'Coach: the next pose')
        try:
            rig = PL.rig_from_project(project, sensor.uid, rec.get('uid') if RC.find_board(project, rec.get('uid'))
                                      else None)
            rig.X, rig.Z = np.asarray(X, float), np.asarray(res.Z[bname], float)
            from ..calib import coach as CO
            taken = [p for p in session.get('poses') or [] if p.get('ids')]
            doc['coach'] = CO.advise(rig, taken, ps, sigma_px=res.sigma_px if math.isfinite(res.sigma_px) else None)
        except (StudyError, ValueError) as e:
            doc['notes'].append(f'The coach could not run: {e}')
    ctx.report(1.0, doc['advice'])
    return doc


def _camera_apply(project, sensor, imager, info: dict, X, sx: dict, ps, intr, label: str) -> dict:
    """The mount pose a calibrated X means for the sensor (its body in its parent frame), as solve_study's 'apply'."""
    from ..calib import planner as PL
    from ..calib import records as RC
    from ..calib.axxb import transform_error
    dt, dr = transform_error(X, info['X0'])
    body = RC.mount_pose_from_X(sensor, imager, X)
    parent, robot_uid = sensor.mount.parent_frame, sensor.mount.robot_uid
    if info['mode'] == 'eye_in_hand':
        parent, robot_uid = info['link'], info['robot_uid']
    elif sensor.mount.on_link:
        parent, robot_uid = 'world', ''
    cal = None if intr is None else {'imager': imager.name, 'record': RC.camera_calibration_from(intr['model'], label)}
    ratio, key = PL.worst(sx, ps.need_mm, ps.need_deg)
    return {'sensor_uid': sensor.uid, 'parent_frame': parent, 'robot_uid': robot_uid, 'pose': body.to_dict(),
            'change_mm': 1000 * dt, 'change_deg': math.degrees(dr), 'calibration': cal, 'ratio': ratio, 'key': key}


def _joint_solve(project, ctx: jobs.Ctx, st: dict) -> dict:
    """Several sessions in one bundle adjustment (module docstring)."""
    from ..calib import capture as CAP
    from ..calib import handeye as HE
    from ..calib import planner as PL
    from ..calib import records as RC
    want = st.get('sessions')
    allss = [s for s in RC.calibration_of(project).get('sessions') or [] if isinstance(s, dict)]
    if want == 'all':
        sessions = [s for s in allss if any(p.get('ids') for p in s.get('poses') or [])]
    else:
        uids = [str(u) for u in (want if isinstance(want, (list, tuple)) else [want])]
        by = {s.get('uid'): s for s in allss}
        missing = [u for u in uids if u not in by]
        if missing:
            raise StudyError(f'{len(missing)} of the sessions chosen are not in the project any more.')
        sessions = [by[u] for u in dict.fromkeys(uids)]
    if len(sessions) < 2:
        raise StudyError('A joint solve needs two or more capture sessions with detections: import another '
                         'session, or solve the one there is on its own.')
    try:
        prob, info = CAP.sessions_problem(project, sessions)
    except CAP.CaptureError as e:
        raise StudyError(str(e)) from None
    if st.get('intrinsics'):
        for c in prob.cameras:
            c.intrinsics = True
    hs = HE.Settings(joint_offsets=tuple(info['robots']) if st.get('joint_offsets') else (),
                     pixel_sigma=float(st['pixel_sigma']) if st.get('pixel_sigma') else None,
                     outlier_k=float(st.get('outlier_k') or HE.OUTLIER_K),
                     outlier_floor_px=float(st.get('outlier_floor_px') or HE.OUTLIER_FLOOR_PX))
    ctx.report(0.05, f'One bundle adjustment over {len(sessions)} sessions')
    res = HE.solve(prob, hs, progress=lambda f, t='': ctx.report(0.05 + 0.75 * f, t))
    rdoc = res.to_dict()
    ps = PL.PlanSettings.from_dict(st)
    names = ' + '.join(s['name'] for s in info['sessions'])
    doc = {'session': {'uid': sessions[0].get('uid'), 'uids': [s.get('uid') for s in sessions], 'name': names,
                       'mode': 'joint', 'poses': sum(len(s.get('poses') or []) for s in sessions),
                       'source': {'kind': 'joint', 'sessions': info['sessions']}, 'stored': True},
           'joint': True, 'ok': res.ok, 'message': res.message, 'result': rdoc,
           'notes': [f'{len(sessions)} sessions solved together: {len(prob.cameras)} camera(s), {len(prob.boards)} '
                     f'board(s), {len(prob.robots)} robot(s), {len(prob.joints)} robot poses.'],
           'requirement': {'need_mm': ps.need_mm, 'need_deg': ps.need_deg,
                           'default': {'need_mm', 'need_deg'} <= set(ps.to_dict()['defaults'])}}
    views = []
    for v in rdoc.get('views') or []:
        si, k = info['views'].get(v.get('name'), (None, None))
        p = (sessions[si].get('poses') or [])[k] if si is not None else {}
        views.append({'name': v.get('name'), 'image': p.get('image', ''), 'session': None if si is None else
                      info['sessions'][si]['name'], 'session_pose': k, 'camera': v.get('camera'),
                      'corners': v.get('n'), 'used': bool(v.get('used')), 'rms_px': v.get('rms_px'),
                      'note': v.get('note', '') or p.get('note', ''),
                      'dt_ms': None if p.get('dt_s') is None else 1000 * float(p['dt_s'])})
    doc['views'] = views
    if not res.ok:
        doc.update(status='critical', word='not solved', advice=res.message, apply=None, applies=[], coach=None,
                   cameras=[], boards=[])
        return doc
    cams, applies = [], []
    for cname, ci in info['cameras'].items():
        sensor, imager = ci['sensor'], RC.sensor_imager(ci['sensor'])
        sx = PL.sigma_dict(res.cov_X[cname])
        ap = _camera_apply(project, sensor, imager, ci, res.X[cname], sx, ps, res.intrinsics.get(cname),
                           f'hand-eye joint solve of {names}')
        sev, word = PL.verdict(ap['ratio'])
        cams.append({'name': cname, 'sensor_uid': sensor.uid, 'imager': imager.name, 'sigma': PL._sig_doc(sx),
                     'frame': res.frames.get(cname, ''), 'status': sev, 'word': word, 'mode': ci['mode'],
                     'ratio': PL._r(ap['ratio']), 'key': ap['key'], 'sessions': ci['sessions'], 'apply': ap})
        applies.append(ap)
    boards = []
    for bname, bi in info['boards'].items():
        rec = bi['record']
        boards.append({'name': bname, 'uid': rec.get('uid', ''), 'frame': res.frames.get(bname, ''),
                       'sigma': PL._sig_doc(PL.sigma_dict(res.cov_Z[bname])),
                       'stored': bool(RC.find_board(project, rec.get('uid'))),
                       'board_pose': {'parent_frame': ((rec.get('mount') or {}).get('parent_frame') or 'world'),
                                      'robot_uid': (rec.get('mount') or {}).get('robot_uid') or '',
                                      'pose': PL._pose(res.Z[bname])}})
    worst = max(cams, key=lambda c: c['ratio'] if c['ratio'] is not None else math.inf)
    order = {'good': 0, 'warning': 1, 'serious': 2, 'critical': 3}
    status = max((c['status'] for c in cams), key=lambda s: order.get(s, 3))
    b0 = next((b for b in boards if b['stored']), boards[0])
    ap0 = dict(worst['apply'], board_uid=b0['uid'] if b0['stored'] else '', board_pose=b0['board_pose'])
    doc.update(status=status, word=next(c['word'] for c in cams if c['status'] == status),
               worst={'ratio': worst['ratio'], 'key': worst['key'], 'camera': worst['name'],
                      'value': PL._r(PL.sigma_dict(res.cov_X[worst['name']]).get(worst['key'], math.inf), 5),
                      'axis': PL.AXIS_WORDS.get((worst['key'] or '').split('_')[0], worst['key'] or '')},
               camera={k: worst[k] for k in ('name', 'sensor_uid', 'imager', 'sigma', 'frame')},
               board={k: b0[k] for k in ('name', 'uid', 'sigma', 'frame')},
               cameras=cams, boards=boards, apply=ap0, applies=applies)
    used = sum(1 for v in views if v['used'])
    moved = '; '.join(f'{c["name"]} {c["apply"]["change_mm"]:.2f} mm, {c["apply"]["change_deg"]:.3f}°' for c in cams)
    doc['advice'] = (f'{used} views of {len(cams)} cameras used, RMS {res.rms_px:.3f} px. From their current mounts: '
                     f'{moved}. Apply writes every calibrated mount.')
    doc['coach'] = None
    if st.get('coach', True):
        ctx.report(0.85, 'Coach: the next pose')
        plannable = [c for c in sorted(cams, key=lambda c: -(c['ratio'] if c['ratio'] is not None else math.inf))
                     if c['mode'] != 'static']
        for c in cams:
            if c['mode'] == 'static':
                doc['notes'].append(f'{c["name"]} and its board are both fixed in the world: it is located through '
                                    'the board, so the coach (which plans robot poses) cannot help it; its σ depends '
                                    'on how much of its image the board fills and on the board\'s own σ.')
        if plannable:
            c = plannable[0]
            si = c['sessions'][0]
            sess, si_info = sessions[si], info['sessions'][si]
            try:
                brec = info['boards'][si_info['board']]['record']
                rig = PL.rig_from_project(project, c['sensor_uid'],
                                          brec.get('uid') if RC.find_board(project, brec.get('uid')) else None)
                rig.X, rig.Z = np.asarray(res.X[c['name']], float), np.asarray(res.Z[si_info['board']], float)
                from ..calib import coach as CO
                taken = [p for p in sess.get('poses') or [] if p.get('ids')]
                doc['coach'] = CO.advise(rig, taken, ps,
                                         sigma_px=res.sigma_px if math.isfinite(res.sigma_px) else None)
                doc['coach'].update(session=si_info['name'], camera=c['name'], robot_uid=si_info['robot_uid'])
            except (StudyError, ValueError, KeyError) as e:
                doc['notes'].append(f'The coach could not run for {c["name"]}: {e}')
    ctx.report(1.0, doc['advice'])
    return doc
