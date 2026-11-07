"""Plan against recording (spec §8.3): a recorded dataset compared with the project's plan, camera by camera: is each
planned camera there, at the planned resolution and frame rate, on the planned mount and in sync, and does it see the
gripper when the plan says it should?

    @jobs.study('plan_vs_recording', 'Plan against recording', quick=False, batch='plan-vs-recording')
                                                                         → --plan-vs-recording OUT.json --dataset PATH
    compare(project, path, audit=None, settings=None, progress=None, cancel=None) -> dict
    run(project, ctx) -> dict

Per dataset camera (matched to the plan's sensors by core/audit/plan.match_cameras: by name, or the dataset entry's
camera mapping in project.data):
- present: the plan's enabled cameras and depth sensors that no dataset camera matches are listed as missing, and
  dataset cameras that match no sensor as not in the plan.
- resolution: the recorded width × height against the plan's stream (the imager core/sim/scene.py records:
  coverage.pick_imager at the sensor's stream): 'same', 'scaled' (the same aspect ratio within ASPECT_TOL, another
  size: the plan's lens scaled), or 'differs' (another aspect ratio: a crop the plan does not model).
- frame rate: the recorded rate (measured from the timestamps by the audit, else the stated one) against the plan's
  (the bandwidth study's stream choice, motion_blur.imaging); 'not stated' when the plan has none. Within
  fps_tol_pct (default 1 %) matches.
- intrinsics, when the dataset stores them (bags' CameraInfo): fx, fy, cx, cy against the plan's lens scaled to the
  recorded size, relative difference.
- mount: stored extrinsics (bags' /tf_static), when present and in the same parent frame, against the plan's optical
  frame (position mm, rotation deg); else the audit's mount estimates: an absolute correction from a board or the
  link silhouette (core/calib/render_calib.estimate), or, from image registration, only the change over the campaign
  (its largest rotation from the first episode, deg) with its classification; 'moved' when the classifier finds a
  step or drift, or the change exceeds mount_tol_deg (default 0.5°; a campaign too short to classify).
- sync: the audit's camera–joint offsets over the episodes (median and range, ms) against max_offset_ms; and against
  the plan's own prediction, the time-sync study's skew interval of the stream (core/studies/time_sync.py: the
  plan's clocks, triggers, exposure, readout and USB transfer; sync_prediction): the recorded median is 'as planned'
  inside the predicted offset widened by 2 σ_δ, else 'outside the prediction' (a comparison of the headline); a
  budget the time-sync study calls incomplete (a term not stated) is a lower bound and gets no verdict.
- visibility: at up to vis_frames frames per episode (default 8) of up to vis_episodes episodes (default 4), the grasp
  study's prediction (core/studies/grasp_visibility.py: its contact pads at the recorded joint state, its Evaluator's
  per-point test: in the image, facing, not hidden in the cube map) against the recorded frame: a camera on the robot
  sees the gripper when the audit's gripper correlation (core/audit/episode.py, 6) is at least gripper_ncc; a static
  camera when at least seen_frac (default 30 %) of the plan's gripper mask pixels differ from the episode's median
  image by more than FG_DN grey levels (the gripper moves over the background). A frame is predicted visible when at
  least pred_frac (default 10 %) of the contact points are visible: the recorded evidence is about the gripper as a
  whole, so the prediction asks whether its pads are in sight at all, not whether most of them are.
  Agreement = (both seen + both hidden) / frames.

Headline: the share of the comparisons (present, resolution, frame rate, mount, sync, sync against the plan when the
plan's budget is complete, visibility per camera) that match the plan.

Known limits: the gripper-visible test from images is evidence, not segmentation (a static camera's median image
holds the gripper wherever it rests most of the episode); registration gives the mount change over the campaign, not
its error against the plan; a dataset without stored extrinsics and without a board or a silhouette fit cannot say
whether the first episode's mount was the planned one; one robot (the primary) is posed.
"""
from __future__ import annotations

import math

import numpy as np

from .. import jobs
from ..jobs import StudyError

SID = 'plan_vs_recording'
DEFAULTS = {'fps_tol_pct': 1.0, 'vis_frames': 8, 'vis_episodes': 4, 'seen_frac': 0.3, 'pred_frac': 0.1,
            'mount_tol_deg': 0.5, 'mount_tol_mm': 5.0}
ASPECT_TOL = 0.01
FG_DN = 25.0


def settings_of(project, given: dict | None = None) -> dict:
    st = dict(DEFAULTS)
    stored = dict(project.analysis.studies.get(SID) or {})
    used = []
    for k, v in dict(stored, **(given or {})).items():
        if k in DEFAULTS and v is not None:
            try:
                st[k] = type(DEFAULTS[k])(v)
            except (TypeError, ValueError):
                raise StudyError(f'The setting {k} must be a number; it is {v!r}.') from None
            used.append(k)
    st['defaults'] = [k for k in DEFAULTS if k not in used]
    return st


def _plan_fps(project) -> dict:
    from . import motion_blur as MB
    try:
        im = MB.imaging(project, MB.settings_of(project))
    except Exception:                                  # noqa: BLE001 — no frame rate then, said per camera
        return {}
    out = {}
    for x in im:
        if x.get('fps') and x['sensor'].uid not in out:
            out[x['sensor'].uid] = float(x['fps'])
    return out


def _plan_sync(project) -> dict:
    """The time-sync study's prediction for the plan ({'streams': [...]}), or {'why': sentence} when it cannot run."""
    try:
        doc = jobs.run_study('time_sync', project.snapshot())
    except StudyError as e:
        return {'why': f'The time-sync study cannot predict the skew: {e}'}
    return {'streams': doc.get('streams') or []}


def sync_prediction(pred: dict, uid: str, imager: str, offsets: list) -> dict:
    """The recorded camera-joint offsets of a camera against the time-sync study's predicted skew of its stream.

    The audit's offset is the frames' content time minus their stamp, relative to the joint states (core/audit
    episode.py, 5); the time-sync study's skew is the stamp minus the exposure instant, relative to joint_states'
    own stamp error (core/studies/time_sync.py). So the predicted offset is the negated skew interval,
    [−skew_high, −skew_low]. The recorded median is as planned when it lies in that interval widened by twice the
    audit's typical σ_δ (the median over the episodes). A budget with terms not stated (time sync's 'incomplete') is
    a lower bound on the delay: only an offset later than its high end contradicts it, so no verdict is given;
    the figures are shown."""
    if 'why' in pred:
        return {'verdict': 'not predicted', 'why': pred['why'], 'as_planned': None}
    st = [x for x in pred.get('streams') or [] if x.get('sensor_uid') == uid]
    st = next((x for x in st if x.get('imager') == imager), st[0] if st else None)
    if st is None:
        return {'verdict': 'not predicted', 'why': 'The time-sync study has no stream for this camera.',
                'as_planned': None}
    lo, hi = -1000.0 * float(st['skew_high_s']), -1000.0 * float(st['skew_low_s'])
    out = {'predicted_offset_ms': [lo, hi], 'complete': bool(st.get('complete')), 'stream': st.get('name'),
           'clock': st.get('clock'), 'trigger': st.get('trigger')}
    offs = [float(o['offset_ms']) for o in offsets]
    if not offs:
        return {**out, 'verdict': 'not measured', 'as_planned': None}
    med = float(np.median(offs))
    sig = [float(o['sigma_ms']) for o in offsets if o.get('sigma_ms') is not None]
    tol = 2.0 * float(np.median(sig)) if sig else 0.0
    out.update(median_ms=med, tolerance_ms=tol)
    if not out['complete']:
        later = med > hi + tol
        return {**out, 'as_planned': None, 'verdict': 'stamps earlier than the lower bound allows' if later else
                'within the lower bound (the budget is incomplete)'}
    ok = lo - tol <= med <= hi + tol
    return {**out, 'as_planned': ok, 'verdict': 'as planned' if ok else 'outside the prediction'}


def _rot_deg(R) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))


def _visibility(project, ds, audit, cam_map, plan, st, cancel=None) -> dict:
    """{camera key: {'frames', 'both_seen', 'predicted_only', 'recorded_only', 'both_hidden', 'agreement'}}."""
    from ..audit import campaign as AC
    from . import grasp_visibility as GV
    out = {}
    robot = project.primary_robot
    if robot is None or plan is None or plan.model is None:
        return {'note': 'The plan has no robot to pose.'}
    notes = []
    try:
        gst = GV.settings_of(project)
        ev = GV.Evaluator(project, gst, notes)
        pads = GV.snapped_pads(project, robot.uid, gst['snap_mm'] / 1000.0, notes)
    except StudyError as e:
        return {'note': f'The grasp study cannot predict visibility here: {e}'}
    by_uid = {c.uid: j for j, c in enumerate(ev.cams)}
    keys = [k for k, u in cam_map.items() if u in by_uid]
    eps = [e for e in audit['episodes']][:st['vis_episodes']]
    req_ncc = (audit.get('requirements_used') or {}).get('gripper_ncc', 0.5)
    for key in keys:
        out[key] = {'frames': 0, 'both_seen': 0, 'predicted_only': 0, 'recorded_only': 0, 'both_hidden': 0}
    for e in eps:
        if cancel is not None and cancel():
            break
        ep = ds.load(e['index'])
        for key in keys:
            cs = ep.cameras.get(key)
            cam = e['cameras'].get(key)
            if cs is None or cam is None or not cs.n_frames:
                continue
            uid = cam_map[key]
            j = by_uid[uid]
            pc = plan.camera(key, int(cs.width or cam['size'][0]), int(cs.height or cam['size'][1]), uid)
            idx = np.linspace(0, cs.n_frames - 1, min(st['vis_frames'], cs.n_frames)).round().astype(int).tolist()
            ts = np.asarray(cs.timestamps, float)
            from ..audit.episode import _grey, _interp_q
            Q = _interp_q(ep, ts[idx])
            seen_rec = {}
            if pc.on_link:
                ncc = ((cam.get('per_frame') or {}).get('gripper_ncc')) or []
                for i in idx:
                    v = ncc[i] if i < len(ncc) else None
                    seen_rec[i] = None if v is None else bool(v >= req_ncc)
            else:
                si = AC.static_image(cs, 60, 320)
                if si is None:
                    continue
                bg = si[0]
                frames = dict(cs.frames(idx))
                for n, i in enumerate(idx):
                    m = pc.gripper_mask(Q[n])
                    if m is None or i not in frames:
                        seen_rec[i] = None
                        continue
                    g = _grey(np.asarray(frames[i]), 320).astype(np.float32)
                    import cv2
                    mm = cv2.resize(m.astype(np.uint8), (g.shape[1], g.shape[0]),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
                    if mm.sum() < 5:
                        seen_rec[i] = False
                        continue
                    seen_rec[i] = bool(np.mean(np.abs(g[mm] - bg[mm]) > FG_DN) >= st['seen_frac'])
            qm = plan.q_model(Q)
            for n, i in enumerate(idx):
                if seen_rec.get(i) is None:
                    continue
                qd = {robot.uid: qm[n]}
                fkd = ev.fk(qd)
                P, N, _seg = GV.pad_world(pads, fkd[robot.uid])
                inimg, facing, unocc, _front = ev.states(j, fkd, qd, P, N, None, False)
                pred = bool(np.mean(inimg & facing & unocc) >= st['pred_frac'])
                rec = seen_rec[i]
                d = out[key]
                d['frames'] += 1
                d['both_seen' if pred and rec else 'predicted_only' if pred else 'recorded_only' if rec
                  else 'both_hidden'] += 1
    for d in out.values():
        if isinstance(d, dict) and d.get('frames'):
            d['agreement'] = (d['both_seen'] + d['both_hidden']) / d['frames']
    if notes:
        out['notes'] = notes[:5]
    return out


def compare(project, path, audit: dict | None = None, settings: dict | None = None, progress=None,
            cancel=None) -> dict:
    """The dataset at path against the project's plan (module docstring)."""
    from ..audit import campaign as AC
    from ..audit import plan as AP
    from ..datasets import open_dataset
    st = settings_of(project, settings)
    report = progress or (lambda f, t='': None)
    if audit is None:
        report(0.0, 'auditing the dataset')
        audit = AC.audit_dataset(path, project, progress=lambda f, t='': report(0.7 * f, t), cancel=cancel)
    plan_fps = _plan_fps(project)
    predicted_sync = _plan_sync(project)
    with open_dataset(path) as ds:
        refs = ds.episodes()
        if not refs:
            raise StudyError('The dataset has no episodes.')
        ep0 = ds.load(refs[0].index)
        plan = AP.PlanContext(project, ep0.joint_names) if project.robots else None
        cam_map = {k: v['sensor_uid'] for k, v in (audit.get('cameras') or {}).items()}
        sensors = {s.uid: s for s in project.variant.sensors if s.enabled and s.kind in ('camera', 'depth')}
        rows, checks = [], []
        for key, cs in ep0.cameras.items():
            uid = cam_map.get(key)
            row = {'key': key, 'sensor_uid': uid, 'sensor': sensors[uid].name if uid in sensors else None,
                   'matched': (audit.get('cameras') or {}).get(key, {}).get('matched')}
            if uid not in sensors or plan is None:
                row['present'] = 'not in the plan'
                rows.append(row)
                checks.append(('present', key, False))
                continue
            row['present'] = 'yes'
            checks.append(('present', key, True))
            W, H = int(cs.width or 0), int(cs.height or 0)
            pc = plan.camera(key, W or 1, H or 1, uid)
            pw, ph = pc.stream
            if (W, H) == (pw, ph):
                res = 'same'
            elif W and H and abs(W / H - pw / ph) <= ASPECT_TOL * pw / ph:
                res = 'scaled'
            else:
                res = 'differs'
            row['resolution'] = {'recorded': [W, H], 'plan': [pw, ph], 'verdict': res}
            checks.append(('resolution', key, res != 'differs'))
            meas = [c['timing'].get('fps_measured') for e in audit['episodes'] for k_, c in e['cameras'].items()
                    if k_ == key and c['timing'].get('fps_measured')]
            fr = float(np.median(meas)) if meas else (cs.fps or None)
            pf = plan_fps.get(uid)
            ok = None if (pf is None or fr is None) else abs(fr - pf) / pf * 100 <= st['fps_tol_pct']
            row['fps'] = {'recorded': fr, 'plan': pf, 'verdict': 'not stated by the plan' if pf is None else
                          ('matches' if ok else 'differs')}
            if ok is not None:
                checks.append(('fps', key, ok))
            if cs.intrinsics and cs.intrinsics.get('K'):
                K = np.asarray(cs.intrinsics['K'], float)
                c = pc.cam
                rel = [abs(K[0, 0] - c.fx) / c.fx, abs(K[1, 1] - c.fy) / c.fy]
                row['intrinsics'] = {'fx_rel': rel[0], 'fy_rel': rel[1], 'cx_px': float(K[0, 2] - c.cx),
                                     'cy_px': float(K[1, 2] - c.cy)}
            mount = {}
            ex = cs.extrinsics
            T_plan = pc.T_po if pc.on_link else pc.T_static
            parent = sensors[uid].mount.parent_frame if pc.on_link else 'world'
            if ex and ex.get('pose') and str(ex.get('frame')) in (parent, 'world' if not pc.on_link else parent):
                from ..frames import Pose
                T_rec = Pose.from_dict(ex['pose']).matrix()
                D = np.linalg.inv(T_plan) @ T_rec
                mount = {'source': 'stored extrinsics', 'position_mm': float(np.linalg.norm(D[:3, 3]) * 1000),
                         'rotation_deg': _rot_deg(D[:3, :3])}
                mount['verdict'] = 'matches' if (mount['position_mm'] <= st['mount_tol_mm'] and
                                                 mount['rotation_deg'] <= st['mount_tol_deg']) else 'differs'
            else:
                m = (audit.get('mounts') or {}).get(key) or {}
                S = m.get('episodes') or []
                if S:
                    absolute = all(s.get('absolute') for s in S)
                    R = np.array([[0.0 if v is None else v for v in s['rotation_deg']] for s in S], float)
                    change = R - R[0]
                    k = int(np.argmax(np.linalg.norm(change, axis=1)))
                    mount = {'source': m.get('source'), 'absolute': absolute, 'kind': m.get('kind'),
                             'change_deg': change[k].tolist(), 'change_episode': S[k]['episode'],
                             'change_norm_deg': float(np.linalg.norm(change[k]))}
                    if absolute:
                        mount['rotation_deg'] = float(np.linalg.norm(R[0]))
                    moved = m.get('kind') in ('slip', 'gradual', 'excursion') or \
                        mount['change_norm_deg'] > st['mount_tol_deg']
                    mount['verdict'] = 'moved' if moved else 'steady'
                else:
                    mount = {'source': None, 'verdict': 'not measured'}
            row['mount'] = mount
            if mount.get('verdict') in ('matches', 'differs', 'moved', 'steady'):
                checks.append(('mount', key, mount['verdict'] in ('matches', 'steady')))
            offrows = [o for o in (audit.get('offsets') or {}).get(key, []) if o.get('offset_ms') is not None]
            offs = [o['offset_ms'] for o in offrows]
            lim = (audit.get('requirements_used') or {}).get('max_offset_ms', 25.0)
            sync_fail = any(r['id'] == 'max_offset_ms' and r['camera'] == key and not r['pass']
                            for e in audit['episodes'] for r in e['requirements'])
            row['sync'] = {'median_ms': float(np.median(offs)) if offs else None,
                           'range_ms': [float(min(offs)), float(max(offs))] if offs else None, 'limit_ms': lim,
                           'verdict': 'not measured' if not offs else ('out of sync' if sync_fail else 'in sync')}
            if offs:
                checks.append(('sync', key, not sync_fail))
            pr = sync_prediction(predicted_sync, uid, pc.imager.name, offrows)
            row['sync']['plan'] = pr
            if pr.get('as_planned') is not None:
                checks.append(('sync against the plan', key, pr['as_planned']))
            rows.append(row)
        missing = [s.name for u, s in sensors.items() if u not in cam_map.values()]
        for name in missing:
            checks.append(('present', name, False))
        report(0.8, 'visibility against the grasp study')
        vis = _visibility(project, ds, audit, cam_map, plan, st, cancel)
        for row in rows:
            v = vis.get(row['key']) if isinstance(vis, dict) else None
            if isinstance(v, dict) and v.get('frames'):
                row['visibility'] = v
                checks.append(('visibility', row['key'], v['agreement'] >= 0.8))
    n = len(checks)
    good = sum(1 for c in checks if c[2])
    pct = 100.0 * good / n if n else None
    sev = 'good' if n and good == n else 'warning' if pct and pct >= 80 else 'serious' if pct and pct >= 50 \
        else 'critical'
    word = {'good': 'matches the plan', 'warning': 'mostly matches', 'serious': 'differs',
            'critical': 'does not match'}[sev]
    bad = [c for c in checks if not c[2]]
    advice = ('The recording matches the plan in every comparison.' if not bad else
              'Differs: ' + '; '.join(f'{c[0]} of {c[1].replace("observation.images.", "")}' for c in bad[:6]) + '.')
    report(1.0, 'done')
    return {'summary': {'checks': n, 'match': good, 'match_pct': pct, 'status': sev, 'word': word, 'advice': advice},
            'cameras': rows, 'missing': missing, 'checks': [{'what': c[0], 'camera': c[1], 'match': c[2]}
                                                            for c in checks],
            'visibility_notes': vis.get('notes') if isinstance(vis, dict) else None,
            'audit_summary': audit.get('summary'), 'dataset': audit.get('dataset'), 'settings': st}


@jobs.study(SID, 'Plan against recording', quick=False, batch='plan-vs-recording', needs_robot=True)
def run(project, ctx: jobs.Ctx) -> dict:
    """The dataset against the plan (module docstring)."""
    from . import audit as AU
    path = AU.dataset_path_for(project, SID)
    st = dict(project.analysis.studies.get(SID) or {})
    doc = compare(project, path, None, {k: st[k] for k in DEFAULTS if k in st}, progress=ctx.report,
                  cancel=ctx.cancelled)
    doc['fingerprint'] = st.get('fingerprint')
    return doc
