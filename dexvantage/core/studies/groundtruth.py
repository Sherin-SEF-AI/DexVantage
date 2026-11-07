"""Rendered against predicted (spec §9.1, MuJoCo ground truth): the grasp study's predicted fingertip visibility and
the depth-validity study's predicted valid region, compared with MuJoCo renders of the same scene.

    @jobs.study('groundtruth', 'Rendered against predicted', batch='groundtruth', needs_robot=True)
                                                                            → --groundtruth OUT.json
    run(project, ctx) -> dict          the study (also core.jobs.run_study('groundtruth', project))
    settings_of(project) -> dict       the settings with the labelled defaults filled in
    status_for(pct, st)                the headline's status and word for an agreement (%)
    decode_thumb(t) -> dict            a disagreement thumbnail's images as numpy arrays

The scene is core/sim/scene.py's (the robots, the task objects, the surfaces, one MuJoCo camera per imager with its
intrinsics) and the images core/sim/render.py's (MuJoCo's OpenGL renderer through EGL, the lens model's distortion
applied by remapping): segmentation per robot link, object and surface, and depth along the optical axis.

(a) Contact visibility. A sample of the project's grasp set (settings grasps, default 50 grasps, a default; drawn
without replacement with the seed rng_seed, default 0; the whole set when it is smaller), the robot's home pose, and
trajectory_frames (default 8, a default) evenly spaced frames, both ends included, of each of that robot's joint
trajectories (project.tasks.trajectories, posed by core/robot/trajectory.py on_model: the joints a trajectory does not
name held at the grasp study's joint state); a project with trajectories but no grasp set compares home and those
frames. Every grasp is solved and evaluated by the grasp study itself (core/studies/grasp_visibility.py: its
settings, its IK with the obstacle check (solve_grasps), its contact patches (snapped_pads, pad_world) and its per-point
visibility, Evaluator.states, evaluated at each grasp's own joint vector without the q-bucket cache): a contact point
is PREDICTED visible from a camera when it projects inside the image and nothing hides the contact location (the
point lifted lift_mm, default 2 mm, along the pad's normal) in the camera's cube map. The same camera (the imager the
grasp study uses, at its stream size) is rendered at the same joint state, and a contact point is RENDERED visible
when it projects inside the image, the segmentation at its pixel (the pixel whose centre is nearest its projection) is
the fingertip's own link, and the rendered depth there agrees with the point's depth within depth_tol_mm (default
5 mm, a default: the lift and an oblique pad seen through a pixel's offset). Points that project outside an image are
counted apart (both sides agree on them by construction); the agreement and the confusion matrix are over the
(camera, point) pairs in view. Reported: per case (home, each grasp, each trajectory frame: the table's 'kind'), per
kind ('by_kind') and in total the confusion matrix (true positive: both
visible; false positive: predicted visible, rendered not; false negative: predicted hidden, rendered visible; true
negative: both hidden), the agreement (TP + TN over all) and the IoU of the visible-point sets (TP / (TP + FP +
FN)); per camera; and the grasp-level confusion of the grasp study's requirement (its own grasp_score on the
predicted and on the rendered fingertip fractions, over the grasps only). Every disagreeing point gets a reason:
'edge' (its pixel is within one pixel of a segmentation boundary: the pixel grid decides it), 'definition' (the render's
occlusion test at the contact location, the rendered range along the lifted point's pixel at least its distance less
eps_mm, agrees with the prediction: the two rules differ, typically a pad seen edge-on, whose lifted contact location is
in view while the pad face itself is not) or 'occlusion' (the cube map and the render disagree about what is in front).

(b) Depth validity. For every depth sensor the depth-validity study can analyse (core/studies/depth_validity.py setup:
its depth resolution, stereo pair, min-Z, range, projector and materials), at the home pose, the reference imager and
the other imagers the depth needs are rendered (depth along the optical axis and the range along each pixel's ray).
The reference depth gets the sensor's noise model: z̃ = z + σ_z(z)·n per pixel, n ~ N(0, 1) seeded, σ_z = z² σ_d /
(f B) (the study's sigma_z with its σ_d and baseline; no noise for a technology without that model, said so).
- On the study's own voxels (the headline figure): the voxels of the task volumes (voxels.SparseGrid, those inside a
  solid object left out, as the study does). PREDICTED: the study's evaluate (its fields of view, min-Z, range, cube-map
  occlusion from every imager, projector cone, materials). RENDERED: the same field-of-view, min-Z, range, projector
  and material conditions (geometric functions of the voxel, the same on both sides), and the occlusion from the
  renders: a voxel is hidden from the reference imager when the noisy depth at its pixel is nearer than the voxel by
  more than eps (occlusion.EPS_M, 1 mm, the study's), and from another imager when the rendered range at its pixel is
  nearer than the voxel by more than eps. What is compared is therefore the study's occlusion (finger shadows, stereo
  parallax) against the render, with the noise moving voxels within a few σ_z of a surface.
- On the surfaces the reference imager sees (reported beside it): every depth_stride-th rendered pixel (default 2)
  that sees a surface gives a point X. PREDICTED: evaluate at X. RENDERED: X̃ from the noisy depth on the pixel's ray
  is valid when z̃ is within min-Z and range, X̃ lies in every needed imager's image, the other imagers' rendered range
  at X̃'s pixel agrees with X̃'s distance within depth_lr_tol_mm (default 5 mm, a default) plus 3 σ_z (a left-right
  consistency check, as stereo matchers make), and the projector and material conditions hold. A point exactly on a
  surface is where a cube-map texel's 1 mm tolerance is weakest, so this figure is lower; it is shown, not hidden.
MuJoCo renders neither an IR pattern nor IR materials, so the projector and material conditions are the study's own on
both sides. Reported per sensor: agreement, IoU of the valid sets, the confusion matrix, the disagreements by the reason
each side gives, and for the surfaces the same inside the task volumes.

Status (headline, the contact agreement, else the depth agreement): 'good' (agrees) from good_pct (default 90 %),
'warning' (mostly agrees) from warn_pct (default 75 %), else 'critical' (disagrees): defaults, not standards.

Settings (project.analysis.studies['groundtruth'], all optional): grasps (50), rng_seed (0), depth_tol_mm (5),
depth_lr_tol_mm (5), depth_stride (2), contacts (true), depth (true), thumbs (4), thumb_px (320), good_pct (90),
warn_pct (75), device ('auto' or an EGL device index), chunk (views per render call, 8), trajectory_frames (8; 0
leaves the trajectories out). The grasp study's own
settings (project.analysis.studies['grasp_visibility']: its grasp set, cameras, lift_mm, eps_mm, cube_n, ...) apply
to the prediction.

Known limits: the prediction leaves out the camera's seat on its mount link (scenegeom's 25 mm rule) and the render
does not; MuJoCo draws cylinders and spheres as smooth primitives while the predictions tessellate them (24 segments);
the rendered rule samples one pixel per contact point, so a point within about a pixel of a silhouette can go either
way (counted as 'edge'); the noise model is first order in disparity and Gaussian (no matcher bias, no holes from
texture, no multipath); the projector pattern and IR materials are not rendered (the study's own rules on both sides);
depth validity is compared at the home pose only, one pixel per voxel; renders are geometric (MuJoCo's shading, no
exposure or blur); a lens rendered through cube faces moves a lot of pixels between processes (slow, about one view
per second at 1920 × 1080).
"""
from __future__ import annotations

import math
import time

import numpy as np

from .. import jobs
from ..jobs import StudyError

SID = 'groundtruth'
DEFAULTS = {
    'grasps': 50,                 # default sample of the grasp set
    'rng_seed': 0,
    'depth_tol_mm': 5.0,          # default: rendered depth agrees with a contact point's depth within this
    'depth_lr_tol_mm': 5.0,       # default: left-right consistency tolerance of the rendered stereo depth
    'depth_stride': 2,            # default: every 2nd pixel of the reference image
    'contacts': True,
    'depth': True,
    'thumbs': 4,
    'thumb_px': 320,
    'good_pct': 90.0,             # default status thresholds (not standards)
    'warn_pct': 75.0,
    'device': 'auto',
    'chunk': 8,
    'trajectory_frames': 8,       # default: evenly spaced frames of each trajectory of the grasp set's robot
}
WORDS = {'good': 'agrees', 'warning': 'mostly agrees', 'critical': 'disagrees'}
WHY = ('edge', 'definition', 'occlusion')
WHY_TEXT = {'edge': 'within a pixel of a segmentation edge (the pixel grid decides it)',
            'definition': 'the occlusion tests agree; the rules differ (the prediction tests the contact location '
                          'in front of the pad, the render the pad surface itself: a pad seen edge-on)',
            'occlusion': 'the cube map and the render disagree about what is in front'}
DEPTH_REASONS = ('fov', 'min_z', 'max_range', 'occluded', 'projector', 'material')
ST_OUT, ST_TP, ST_FP, ST_FN, ST_TN = 0, 1, 2, 3, 4      # per contact point and camera (decode)
STATE_TEXT = {ST_OUT: 'outside the image', ST_TP: 'visible in both', ST_FP: 'predicted visible, hidden in the render',
              ST_FN: 'predicted hidden, visible in the render', ST_TN: 'hidden in both'}


# ====================================================================== settings
def settings_of(project) -> dict:
    given = dict(project.analysis.studies.get(SID) or {})
    out = dict(DEFAULTS)
    used = []
    for k in DEFAULTS:
        if given.get(k) is not None:
            out[k] = given[k]
        else:
            used.append(k)
    for k in ('depth_tol_mm', 'depth_lr_tol_mm', 'good_pct', 'warn_pct'):
        try:
            out[k] = float(out[k])
        except (TypeError, ValueError):
            raise StudyError(f'The setting {k} must be a number; it is {out[k]!r}.') from None
        if not math.isfinite(out[k]) or out[k] < 0:
            raise StudyError(f'The setting {k} must be zero or a positive number; it is {out[k]:g}.')
    for k in ('grasps', 'rng_seed', 'depth_stride', 'thumbs', 'thumb_px', 'chunk', 'trajectory_frames'):
        try:
            out[k] = int(out[k])
        except (TypeError, ValueError):
            raise StudyError(f'The setting {k} must be a whole number; it is {out[k]!r}.') from None
        if out[k] < 0:
            raise StudyError(f'The setting {k} must not be negative; it is {out[k]}.')
    out['grasps'] = max(1, out['grasps'])
    out['depth_stride'] = max(1, out['depth_stride'])
    out['chunk'] = max(1, out['chunk'])
    out['thumb_px'] = min(max(out['thumb_px'], 64), 1280)
    out['contacts'] = bool(out['contacts'])
    out['depth'] = bool(out['depth'])
    out['defaults'] = used
    return out


def status_for(pct, st: dict) -> tuple[str, str]:
    if pct is None:
        return 'warning', 'nothing compared'
    sev = 'good' if pct >= st['good_pct'] else 'warning' if pct >= st['warn_pct'] else 'critical'
    return sev, WORDS[sev]


def _num(x, nd=6):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def _confusion(pred: np.ndarray, rend: np.ndarray) -> dict:
    p, r = np.asarray(pred, bool), np.asarray(rend, bool)
    tp, fp = int((p & r).sum()), int((p & ~r).sum())
    fn, tn = int((~p & r).sum()), int((~p & ~r).sum())
    n = tp + fp + fn + tn
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'n': n,
            'agreement': _num((tp + tn) / n) if n else None,
            'iou': _num(tp / (tp + fp + fn)) if tp + fp + fn else None}


def _add(a: dict, b: dict) -> dict:
    out = {k: a.get(k, 0) + b.get(k, 0) for k in ('tp', 'fp', 'fn', 'tn', 'n')}
    n = out['n']
    out['agreement'] = _num((out['tp'] + out['tn']) / n) if n else None
    d = out['tp'] + out['fp'] + out['fn']
    out['iou'] = _num(out['tp'] / d) if d else None
    return out


def _pix(uv, W, H):
    u = np.clip(np.rint(uv[:, 0]), 0, W - 1).astype(np.int64)
    v = np.clip(np.rint(uv[:, 1]), 0, H - 1).astype(np.int64)
    return u, v


def _edge(seg: np.ndarray, u, v) -> np.ndarray:
    """(K,) True where the pixel's 3 × 3 neighbourhood holds more than one segmentation value."""
    H, W = seg.shape
    s0 = seg[v, u]
    out = np.zeros(len(u), bool)
    for dv in (-1, 0, 1):
        for du in (-1, 0, 1):
            if du or dv:
                out |= seg[np.clip(v + dv, 0, H - 1), np.clip(u + du, 0, W - 1)] != s0
    return out


# ====================================================================== (a) contact visibility
def finger_links(model, link: str) -> list:
    """The links rigidly attached to a fingertip's pad link: its MuJoCo body, the bodies welded to it (no joint
    between them) up to the first moving joint and below it, and the links mapped onto those bodies. A gripper's pad
    link often has no visual geometry of its own (the Robotiq 2F-85's pad is a collision box; its visual silicone pad
    and follower are welded to it), so the render's "own link" is this group."""
    m = model.m
    b = int(model._lb[model.link_index(link)])
    root = b
    while root > 0 and int(m.body_jntnum[root]) == 0 and int(m.body_parentid[root]) > 0:
        root = int(m.body_parentid[root])

    def rigid_root(x):
        while x > 0 and int(m.body_jntnum[x]) == 0 and int(m.body_parentid[x]) > 0:
            x = int(m.body_parentid[x])
        return x
    bodies = {x for x in range(1, m.nbody) if rigid_root(x) == root}
    return [n for i, n in enumerate(model.link_names) if int(model._lb[i]) in bodies]


def _contacts(project, st, R, ctx, notes, t_frac) -> dict:
    from .. import grasps as GR
    from . import grasp_visibility as GV
    gst = GV.settings_of(project)
    sets = GR.sets(project)
    trajs = _trajectories(project)
    if not sets and not trajs:
        return {'skipped': 'There is no grasp set or trajectory: make a grasp set on the Tasks page (or add a '
                           'trajectory) to compare the contact visibility.'}
    if not sets:
        return _contacts_poses(project, st, R, ctx, notes, t_frac, gst, None, trajs)
    gs = GR.get_set(project, gst['grasp_set']) if gst['grasp_set'] else sets[0]
    if gs is None:
        return {'skipped': 'The grasp set chosen in the grasp study\'s settings is not in the project any more.'}
    return _contacts_poses(project, st, R, ctx, notes, t_frac, gst, gs, trajs)


def _trajectories(project) -> list:
    """The project's joint trajectories that can be posed (two samples or more, a robot of the project)."""
    return [t for t in project.tasks.trajectories if isinstance(t, dict) and len(t.get('t') or []) >= 2
            and project.robot(t.get('robot_uid', '')) is not None]


def trajectory_cases(trajs: list, robot_uid: str, names, hold, frames: int) -> list:
    """[(case name, trajectory uid, t, q)]: `frames` evenly spaced times (both ends included) of every trajectory of
    the robot, q on the model's joints (the joints a trajectory does not name held at hold)."""
    from ..robot import trajectory as TJ
    out = []
    if frames <= 0:
        return out
    for tr in trajs:
        if tr.get('robot_uid') != robot_uid:
            continue
        t0, t1 = TJ.duration(tr)
        ts = [t0] if frames == 1 or t1 <= t0 else list(np.linspace(t0, t1, frames))
        for t in ts:
            out.append((f'{tr.get("name") or "trajectory"} at {t:.2f} s', str(tr.get('uid', '')), float(t),
                        TJ.on_model(tr, names, hold, float(t))))
    return out


def _contacts_poses(project, st, R, ctx, notes, t_frac, gst, gs, trajs) -> dict:
    from .. import camrender as CR
    from .. import grasps as GR
    from ..frames import Pose
    from . import coverage as CV
    from . import grasp_visibility as GV
    if gs is not None:
        robot = project.robot(gs.get('robot_uid', ''))
        if robot is None:
            return {'skipped': f'The grasp set "{gs.get("name")}" is for a robot that is not in the project.'}
        poses = list(gs.get('poses') or [])
        if not poses and not any(t.get('robot_uid') == robot.uid for t in trajs):
            return {'skipped': f'The grasp set "{gs.get("name")}" has no grasps.'}
    else:
        robot = project.robot(trajs[0].get('robot_uid', ''))
        poses = []
    n = min(st['grasps'], len(poses))
    rng = np.random.default_rng([int(st['rng_seed']), 11])
    sel = np.sort(rng.choice(len(poses), n, replace=False)) if n < len(poses) else np.arange(len(poses))
    gsel = dict(gs or {}, poses=[poses[i] for i in sel])
    ev = GV.Evaluator(project, gst, notes)
    if robot.uid not in ev.robots:
        raise StudyError(f'{robot.name} cannot be posed, so its grasps cannot be compared.')
    if not ev.cams:
        return {'skipped': 'There is no camera or depth sensor in use for the grasp study to compare.'}
    model = ev.robots[robot.uid].model
    qd0, _src = CV.joint_state(project, gst, notes)
    hold = np.asarray(qd0.get(robot.uid, model.home), float)
    pads = GV.snapped_pads(project, robot.uid, gst['snap_mm'] / 1000.0, notes)
    if len(sel):
        try:
            link, tcp_off, targets = GR.ik_targets(project, gsel)
        except GR.GraspError as e:
            raise StudyError(str(e)) from None
        sub = jobs.Ctx(progress=lambda f, t='': ctx.report(t_frac[0] + 0.25 * (t_frac[1] - t_frac[0]) * f, t),
                       cancel=ctx.cancel)
        b = GV.solve_grasps(project, robot, model, link, targets, tcp_off, gst, hold, sub, notes)
        feas = np.asarray(b.ok, bool)
        Q = np.asarray(b.q, float)
    else:
        feas, Q = np.zeros(0, bool), np.zeros((0, len(model.joint_names)))
    others = {u: v for u, v in qd0.items() if u != robot.uid}
    cases = [('home', -1, np.asarray(model.home, float), 'home', None)]
    cases += [(str(gsel['poses'][i].get('id', i)), int(sel[i]), Q[i], 'grasp', None) for i in np.flatnonzero(feas)]
    tcases = trajectory_cases(trajs, robot.uid, model.joint_names, hold, st['trajectory_frames'])
    cases += [(name, -1, q, 'trajectory', {'uid': tu, 't': t}) for name, tu, t, q in tcases]
    F = len(pads)
    groups = [finger_links(model, p['link']) for p in pads]
    own_sets = [np.array([R.owner_index(('robot', robot.uid, ln)) for ln in g], np.int32) for g in groups]
    own_sets = [x[x >= 0] for x in own_sets]
    if any(len(x) == 0 for x in own_sets):
        raise StudyError('The fingertip links have no visual geometry in the MuJoCo scene, so their visibility '
                         'cannot be rendered.')
    lift, eps, tol = gst['lift_mm'] / 1000.0, gst['eps_mm'] / 1000.0, st['depth_tol_mm'] / 1000.0
    C = len(ev.cams)
    # per case: everything the prediction needs, then the views to render
    work = []
    for name, gi, q, kind, tref in cases:
        qd = dict(others)
        qd[robot.uid] = q
        fkd = ev.fk(qd)
        P, N, seg = GV.pad_world(pads, fkd[robot.uid])
        per = []
        for j in range(C):
            inimg, facing, unocc, _front = ev.states(j, fkd, qd, P, N, None, False)
            Rj, oj = ev.cams[j].optical(fkd)
            per.append({'pred': inimg & facing & unocc, 'R': Rj, 'o': oj})
        work.append({'name': name, 'grasp': gi, 'q': q, 'qd': qd, 'P': P, 'N': N, 'seg': seg, 'cams': per,
                     'kind': kind, 'trajectory': tref})
    views = [(w, j) for w in work for j in range(C)]
    totals = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'n': 0}
    cam_tot = [dict(totals) for _ in range(C)]
    why = {k: 0 for k in WHY}
    outside = 0
    dzs = []
    K = len(work[0]['P'])
    states = np.full((len(work), C, K), ST_OUT, np.uint8)
    frac_p = np.full((len(work), C, F), np.nan)
    frac_r = np.full((len(work), C, F), np.nan)
    t0 = time.perf_counter()
    for a in range(0, len(views), st['chunk']):
        if ctx.cancelled():
            break
        chunk = views[a:a + st['chunk']]
        imgs = R.render([{'sensor_uid': ev.cams[j].uid, 'imager': ev.cams[j].imager.name,
                          'camera_model': ev.cams[j].cam, 'q': w['qd'], 'outputs': ('seg', 'depth')}
                         for w, j in chunk])
        for (w, j), im in zip(chunk, imgs):
            c, pc = ev.cams[j].cam, w['cams'][j]
            Rj, oj = pc['R'], pc['o']
            Pc = (w['P'] - oj) @ Rj
            uv, ok = c.project(Pc)
            ok = np.asarray(ok, bool)
            u, v = _pix(np.nan_to_num(uv), c.width, c.height)
            sg, dp, rg = im['seg'], im['depth'], im['range']
            px = sg[v, u]
            mine = np.zeros(len(px), bool)
            for f in range(F):
                at = w['seg'] == f
                mine[at] = np.isin(px[at], own_sets[f])
            dz = dp[v, u] - Pc[:, 2]
            rend = ok & mine & (np.abs(dz) <= tol)
            dzs.append(dz[ok & mine & (pc['pred'])])
            Pl = w['P'] + lift * w['N']
            Plc = (Pl - oj) @ Rj
            uvl, okl = c.project(Plc)
            ul, vl = _pix(np.nan_to_num(uvl), c.width, c.height)
            rl = rg[vl, ul]
            free = np.asarray(okl, bool) & ((rl <= 0) | (rl >= np.linalg.norm(Plc, axis=1) - eps))
            pred = pc['pred']
            pc['rend'], pc['ok'] = rend, ok
            stt = np.where(pred, np.where(rend, ST_TP, ST_FP), np.where(rend, ST_FN, ST_TN)).astype(np.uint8)
            states[work.index(w), j] = np.where(ok, stt, ST_OUT)
            m = ok
            outside += int((~ok).sum())
            cm = _confusion(pred[m], rend[m])
            pc['conf'] = cm
            dis = m & (pred != rend)
            edge = _edge(sg, u, v) & dis
            defn = dis & ~edge & (pred == free)
            occ = dis & ~edge & ~defn
            pc['why'] = {'edge': int(edge.sum()), 'definition': int(defn.sum()), 'occlusion': int(occ.sum())}
            pc['dis_mask'] = dis
            for k in WHY:
                why[k] += pc['why'][k]
            cam_tot[j] = _add(cam_tot[j], cm)
            totals = _add(totals, cm)
            k_case = work.index(w)
            cnt = np.bincount(w['seg'], minlength=F).astype(float)
            frac_p[k_case, j] = np.bincount(w['seg'], weights=pred.astype(float), minlength=F) / np.maximum(cnt, 1)
            frac_r[k_case, j] = np.bincount(w['seg'], weights=rend.astype(float), minlength=F) / np.maximum(cnt, 1)
        ctx.report(t_frac[0] + (t_frac[1] - t_frac[0]) * (0.25 + 0.6 * min(1.0, (a + len(chunk)) / len(views))),
                   f'Rendered {min(a + len(chunk), len(views))} of {len(views)} views')
    render_s = time.perf_counter() - t0
    # the grasp study's requirement on the predicted and the rendered fractions
    sc_p = GV.grasp_score(frac_p, np.full((len(work), C), np.nan), gst)
    sc_r = GV.grasp_score(frac_r, np.full((len(work), C), np.nan), gst)
    meets_p = np.isfinite(sc_p) & (sc_p >= gst['min_visible'])
    meets_r = np.isfinite(sc_r) & (sc_r >= gst['min_visible'])
    is_grasp = np.array([w['kind'] == 'grasp' for w in work], bool)
    gl = _confusion(meets_p[is_grasp], meets_r[is_grasp])
    by_kind = {}
    table = []
    for k, w in enumerate(work):
        cm = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'n': 0}
        wy = {x: 0 for x in WHY}
        cams = []
        for j, pc in enumerate(w['cams']):
            if 'conf' not in pc:
                continue
            cm = _add(cm, pc['conf'])
            for x in WHY:
                wy[x] += pc['why'][x]
            cams.append({'sensor_uid': ev.cams[j].uid, **pc['conf'],
                         'fingertips_predicted': [_num(x, 4) for x in frac_p[k, j]],
                         'fingertips_rendered': [_num(x, 4) for x in frac_r[k, j]]})
        table.append({'case': w['name'], 'kind': w['kind'], 'grasp_index': w['grasp'],
                      **cm, 'why': wy, 'meets_predicted': bool(meets_p[k]), 'meets_rendered': bool(meets_r[k]),
                      'score_predicted': _num(sc_p[k], 4), 'score_rendered': _num(sc_r[k], 4), 'cameras': cams,
                      'q': [float(x) for x in w['q']],
                      **({'trajectory_uid': w['trajectory']['uid'], 't': w['trajectory']['t']}
                         if w['trajectory'] else {})})
        by_kind[w['kind']] = _add(by_kind.get(w['kind'], {}), cm)
    # thumbnails of the worst (case, camera) disagreements
    thumbs = []
    worst = sorted(((pc['conf']['fp'] + pc['conf']['fn'], k, j) for k, w in enumerate(work)
                    for j, pc in enumerate(w['cams']) if 'conf' in pc), reverse=True)
    worst = [x for x in worst if x[0] > 0][:st['thumbs']]
    if worst and not ctx.cancelled():
        from .coverage import encode_array
        tv = []
        for _d, k, j in worst:
            c = ev.cams[j].cam
            s = st['thumb_px'] / c.width
            tv.append({'sensor_uid': ev.cams[j].uid, 'imager': ev.cams[j].imager.name, 'camera_model': c,
                       'q': work[k]['qd'], 'outputs': ('rgb',), 'scale': s})
        imgs = R.render(tv)
        for (d, k, j), im, v in zip(worst, imgs, tv):
            w, pc, cobj = work[k], work[k]['cams'][j], ev.cams[j]
            cs = im['model']
            pose = Pose.from_matrix(np.r_[np.c_[pc['R'], pc['o']], [[0, 0, 0, 1]]])
            g = CR.gather(project, w['qd'])
            g = CR.without_seat(g, project, cobj.sensor, [pc['o']])
            pr = CR.render(g, cs, pose, cs.width, cs.height, floor=False)
            uv, _ok, _d = CR.project_world(cs, pose, w['P'])
            ok = pc['ok']
            thumbs.append({'case': w['name'], 'sensor_uid': cobj.uid, 'camera': cobj.name, 'disagree': int(d),
                           'predicted_rgb': encode_array(pr.rgb), 'rendered_rgb': encode_array(im['rgb']),
                           'uv': encode_array(np.round(uv[ok], 2).astype(np.float32)),
                           'pred': encode_array(pc['pred'][ok].astype(np.uint8)),
                           'rend': encode_array(pc['rend'][ok].astype(np.uint8))})
    cam_docs = [{'sensor_uid': c.uid, 'name': c.name, 'imager': c.imager.name, 'mode': [c.cam.width, c.cam.height],
                 'lens': c.cam.model, **cam_tot[j]} for j, c in enumerate(ev.cams)]
    n_traj = sum(1 for w in work if w['kind'] == 'trajectory')
    return {'grasp_set': ({'uid': gs.get('uid'), 'name': gs.get('name'), 'poses': len(poses), 'sampled': int(n),
                           'feasible': int(feas.sum()), 'seed': st['rng_seed']} if gs is not None else None),
            'trajectories': {'frames': n_traj, 'per_trajectory': st['trajectory_frames'],
                             'uids': sorted({w['trajectory']['uid'] for w in work if w['trajectory']})},
            'by_kind': by_kind,
            'robot': {'uid': robot.uid, 'name': robot.name},
            'fingertips': [{'name': p['name'], 'link': p['link'], 'points': p['points']} for p in pads],
            'cases': len(work), 'totals': totals, 'outside_image': outside, 'why': why,
            'why_text': WHY_TEXT, 'cameras': cam_docs, 'table': table, 'grasp_level': gl,
            'rules': {'lift_mm': gst['lift_mm'], 'eps_mm': gst['eps_mm'], 'depth_tol_mm': st['depth_tol_mm'],
                      'cube_n': gst['cube_n'], 'requirement': GV.requirement_text(gst)},
            'thumbs': thumbs, 'seconds': {'render_and_compare': round(render_s, 3)},
            'depth_offset_mm': _offsets(dzs),
            'states': _enc(states), 'points': _enc(np.array([w['P'] for w in work], np.float32)),
            'point_fingertip': _enc(work[0]['seg'].astype(np.uint8)),
            'occlusion': {'backend': ev.backend, 'renderer': ev.renderer}}


def _offsets(dzs) -> dict:
    """Percentiles (mm) of rendered depth − contact point depth where the pixel shows the finger and the prediction
    says visible: how far the rendered surface is from the contact points (a collision pad outside its visual mesh
    shows here)."""
    d = np.concatenate(dzs) if dzs else np.zeros(0)
    d = d[np.isfinite(d)] * 1000.0
    if not len(d):
        return {'n': 0}
    p5, p50, p95 = np.percentile(d, [5, 50, 95])
    return {'n': int(len(d)), 'p5': _num(p5, 3), 'median': _num(p50, 3), 'p95': _num(p95, 3)}


# ====================================================================== (b) depth validity
def _code(conds) -> np.ndarray:
    """The first failing condition per point, as depth_validity.evaluate numbers it (0: valid)."""
    code = np.zeros(len(conds[0]), np.int8)
    for k in range(len(conds) - 1, -1, -1):
        code[~np.asarray(conds[k], bool)] = k + 1
    return code


def _pairs(pc, rc) -> dict:
    """{'predicted reason → rendered reason': count} over the disagreeing points."""
    dis = (pc == 0) != (rc == 0)
    out: dict = {}
    for a, b in zip(pc[dis], rc[dis]):
        key = f'{"valid" if a == 0 else DEPTH_REASONS[a - 1]} → {"valid" if b == 0 else DEPTH_REASONS[b - 1]}'
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda t: -t[1]))


def _depth(project, st, R, ctx, notes) -> dict:
    from .. import occlusion as OC
    from .. import scenegeom as SG
    from .. import voxels as VX
    from .coverage import _inside_objects, encode_array
    from . import depth_validity as DV
    out, skipped = [], []
    sensors = DV.depth_sensors(project)
    if not sensors:
        return {'sensors': [], 'skipped': [], 'message': 'There is no depth sensor in this rig to compare.'}
    dst = DV.settings(project)
    occ = SG.scene_occluders(project, None)
    vols = [v.bounds() for v in project.workspace.volumes]
    grid = VX.SparseGrid.from_project(project) if project.workspace.volumes else None
    if grid is not None:
        keep = ~_inside_objects(project, grid.centers.astype(np.float64))
        Pv = grid.centers.astype(np.float64)[keep]
        Sv = np.asarray(grid.size, float)[keep]
    eps = OC.EPS_M
    for s in sensors:
        if ctx.cancelled():
            break
        try:
            ds = DV.setup(project, s)
            body = SG.sensor_world_pose(project, s, None)
        except Exception as e:                      # noqa: BLE001 — DepthSetupError, a mount that cannot be posed
            skipped.append({'uid': s.uid, 'name': s.name, 'why': str(e)})
            continue
        try:
            imgs = R.render([{'sensor_uid': s.uid, 'imager': im.name, 'camera_model': im.camera, 'q': None,
                              'outputs': ('depth', 'seg')} for im in ds.needed])
        except KeyError as e:
            skipped.append({'uid': s.uid, 'name': s.name, 'why': str(e).strip('"\'')})
            continue
        k_ref = next(i for i, im in enumerate(ds.needed) if im is ds.ref)
        ref, cam = imgs[k_ref], ds.ref.camera
        T_ref = DV.imager_world(body, ds.ref)
        Ts = [DV.imager_world(body, im) for im in ds.needed]
        origins = [T.position for T in Ts]
        occ_s = SG.sensor_occluders(project, s, None, origins=origins, base=occ)
        mats = DV.material_objects(project, ds.technology)
        cubes: dict = {}
        rng = np.random.default_rng([int(st['rng_seed']), 21, len(out)])
        # the depth sensor's noisy measurement of the rendered reference depth (one draw per pixel)
        zr = ref['depth'].astype(float)
        if ds.has_error_model:
            sz_img = DV.sigma_z(zr, ds.f_px, ds.baseline_m, ds.sigma_d_px)
            noise_note = (f'σ_z = z² σ_d / (f B) with σ_d {ds.sigma_d_px:g} px ({ds.sigma_d_note}), f {ds.f_px:.1f} '
                          f'px, B {ds.baseline_m * 1000:.1f} mm')
        else:
            sz_img = np.zeros_like(zr)
            noise_note = 'no noise added: the depth-validity study has no error model for this technology'
        zt_img = np.where(zr > 0, zr + sz_img * rng.standard_normal(zr.shape), 0.0)
        doc = {'uid': s.uid, 'name': s.name, 'technology': ds.technology,
               'resolution': [int(ds.resolution[0]), int(ds.resolution[1])], 'reference': ds.ref.name,
               'imagers': [im.name for im in ds.needed], 'min_z_m': ds.min_z_m, 'max_range_m': ds.max_range_m,
               'noise': noise_note, 'sigma_z_median_m': _num(np.median(sz_img[zr > 0])) if (zr > 0).any() else None}
        # ---- the study's voxels: its predicted valid region against the rendered (noisy) depth
        if grid is not None and len(Pv):
            pv = DV.evaluate(ds, body, Pv, occluders=occ_s, size=Sv, mats=mats, project=project,
                             n=int(dst['cube_n']), backend=str(dst['backend']), cubes=cubes)
            gv = DV.evaluate(ds, body, Pv, occluders=None, size=Sv, mats=mats, project=project)
            vis = np.ones(len(Pv), bool)
            for k, (im, T) in enumerate(zip(ds.needed, Ts)):
                Q = T.inverse().apply(Pv)
                uv, okk = im.camera.project(Q)
                okk = np.asarray(okk, bool)
                u, v = _pix(np.nan_to_num(uv), im.camera.width, im.camera.height)
                if k == k_ref:
                    zs = zt_img[v, u]
                    hid = okk & (zr[v, u] > 0) & (zs < Q[:, 2] - eps)
                else:
                    rk = imgs[k]['range'][v, u]
                    hid = okk & (rk > 0) & (rk < np.linalg.norm(Q, axis=1) - eps)
                vis &= ~hid
            pcode = pv['code']
            rcode = _code([gv['in_fov'], gv['min_z_ok'], gv['max_ok'], vis, gv['proj_ok'], gv['material'] == 0])
            doc['voxels'] = {**_confusion(pcode == 0, rcode == 0), 'count': int(len(Pv)),
                             'voxel_mm': float(grid.voxel_m * 1000), 'predicted_valid': _num((pcode == 0).mean()),
                             'rendered_valid': _num((rcode == 0).mean()), 'disagreements': _pairs(pcode, rcode)}
        # ---- the surfaces the reference imager sees: every depth_stride-th rendered pixel
        stride = st['depth_stride']
        vv, uu = np.mgrid[0:cam.height:stride, 0:cam.width:stride]
        z = zr[vv, uu].ravel()
        hit = z > 0
        rays = cam.unproject(np.stack([uu.ravel(), vv.ravel()], -1).astype(float))[hit]
        z = z[hit]
        zt = zt_img[vv, uu].ravel()[hit]
        sz = sz_img[vv, uu].ravel()[hit]
        X = T_ref.apply(rays * (z / rays[:, 2])[:, None])
        pv = DV.evaluate(ds, body, X, occluders=occ_s, mats=mats, project=project, n=int(dst['cube_n']),
                         backend=str(dst['backend']), cubes=cubes)
        Xt = T_ref.apply(rays * (zt / rays[:, 2])[:, None])
        rv = DV.evaluate(ds, body, Xt, occluders=None, mats=mats, project=project)
        lr = np.ones(len(z), bool)
        lr_tol = st['depth_lr_tol_mm'] / 1000.0 + 3.0 * sz
        for k, (im, T) in enumerate(zip(ds.needed, Ts)):
            if k == k_ref:
                continue
            Q = T.inverse().apply(Xt)
            uv, okk = im.camera.project(Q)
            u, v = _pix(np.nan_to_num(uv), im.camera.width, im.camera.height)
            ro = imgs[k]['range'][v, u]
            lr &= np.asarray(okk, bool) & (ro > 0) & (np.abs(ro - np.linalg.norm(Q, axis=1)) <= lr_tol)
        rcode = _code([rv['in_fov'], rv['min_z_ok'], rv['max_ok'], lr, rv['proj_ok'], rv['material'] == 0])
        pcode = pv['code']
        inside = np.zeros(len(z), bool)
        for lo, hi in vols:
            inside |= ((X >= lo - 1e-9) & (X <= hi + 1e-9)).all(axis=1)
        Hs, Ws = vv.shape
        pm = np.full(Hs * Ws, 255, np.uint8)
        rm_ = np.full(Hs * Ws, 255, np.uint8)
        pm[np.flatnonzero(hit)] = pcode.astype(np.uint8)
        rm_[np.flatnonzero(hit)] = rcode.astype(np.uint8)
        doc['surfaces'] = {**_confusion(pcode == 0, rcode == 0), 'stride': stride, 'pixels': int(hit.sum()),
                           'predicted_valid': _num((pcode == 0).mean()) if len(z) else None,
                           'rendered_valid': _num((rcode == 0).mean()) if len(z) else None,
                           'inside_volumes': _confusion(pcode[inside] == 0, rcode[inside] == 0),
                           'disagreements': _pairs(pcode, rcode)}
        doc['masks'] = {'shape': [int(Hs), int(Ws)], 'predicted': encode_array(pm.reshape(Hs, Ws)),
                        'rendered': encode_array(rm_.reshape(Hs, Ws))}
        main = doc.get('voxels') or doc['surfaces']
        doc.update({k: main[k] for k in ('tp', 'fp', 'fn', 'tn', 'n', 'agreement', 'iou')})
        doc['compared_on'] = 'voxels' if 'voxels' in doc else 'surfaces'
        out.append(doc)
    tot = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'n': 0}
    for x in out:
        tot = _add(tot, x)
    return {'sensors': out, 'skipped': skipped, 'totals': tot, 'reasons': list(DEPTH_REASONS)}


# ====================================================================== the study
@jobs.study(SID, 'Rendered against predicted', quick=False, batch='groundtruth', needs_robot=True)
def run(project, ctx: jobs.Ctx) -> dict:
    """MuJoCo renders against the grasp study's predicted contact visibility and the depth-validity study's predicted
    valid region (module docstring)."""
    from ...validation.mjrender import RenderUnavailable
    from ..sim.render import Renderer
    st = settings_of(project)
    notes: list = []
    t_all = time.perf_counter()
    ctx.report(0.0, 'Building the MuJoCo scene')
    try:
        R = Renderer(project, device=st['device'])
    except RuntimeError as e:
        raise StudyError(f'The MuJoCo scene could not be built: {e}') from None
    try:
        contacts, depth = None, None
        try:
            if st['contacts']:
                contacts = _contacts(project, st, R, ctx, notes, (0.02, 0.8))
            if st['depth'] and not ctx.cancelled():
                ctx.report(0.82, 'Depth validity against rendered depth')
                depth = _depth(project, st, R, ctx, notes)
            info = R.info
        except RenderUnavailable as e:
            raise StudyError(str(e)) from None
        stats = dict(R.stats)
        scene = {'cameras': [{'name': c.name, 'sensor_uid': c.sensor_uid, 'sensor': c.sensor_name, 'imager': c.imager,
                              'body': c.body, 'mode': [c.model.width, c.model.height], 'lens': c.model.model,
                              'primary': c.primary} for c in R.build.cameras.values()],
                 'owners': len(R.build.owners), 'notes': list(R.build.notes), 'defaults': list(R.build.defaults),
                 'geoms': int(R.model.ngeom), 'bodies': int(R.model.nbody)}
    finally:
        R.close()
    c_ok = contacts is not None and 'totals' in contacts and contacts['totals']['n'] > 0
    d_ok = depth is not None and depth.get('totals', {}).get('n', 0) > 0
    if not c_ok and not d_ok:
        why = ' '.join(x for x in ((contacts or {}).get('skipped', ''), (depth or {}).get('message', '')) if x)
        raise StudyError('Nothing could be compared: ' + (why or 'no contact point or depth pixel is in view.'))
    cpct = 100.0 * contacts['totals']['agreement'] if c_ok and contacts['totals']['agreement'] is not None else None
    dpct = 100.0 * depth['totals']['agreement'] if d_ok and depth['totals']['agreement'] is not None else None
    head = cpct if cpct is not None else dpct
    sev, word = status_for(head, st)
    summary = {'agreement_pct': _num(head, 3), 'headline': 'contacts' if cpct is not None else 'depth',
               'contact_agreement_pct': _num(cpct, 3), 'contact_iou': contacts['totals']['iou'] if c_ok else None,
               'depth_agreement_pct': _num(dpct, 3), 'depth_iou': depth['totals']['iou'] if d_ok else None,
               'grasp_level_agreement': contacts['grasp_level']['agreement'] if c_ok else None,
               'status': sev, 'word': word}
    summary['advice'] = advice(summary, contacts if c_ok else None, depth if d_ok else None, st)
    fps = stats.get('frames_per_s')
    return {'project': project.name, 'variant': project.variant.name,
            'settings': {k: v for k, v in st.items()}, 'summary': summary,
            'contacts': contacts, 'depth': depth, 'scene': scene, 'notes': notes,
            'render': {**info, 'frames': stats.get('frames'), 'render_seconds': _num(stats.get('render_seconds'), 4),
                       'frames_per_s': _num(fps, 2), 'calls': stats.get('calls'),
                       'call_seconds': _num(stats.get('seconds'), 3)},
            'timings': {'total_seconds': round(time.perf_counter() - t_all, 3)}}


def advice(s: dict, contacts, depth, st) -> str:
    """The headline's sentence: what agrees, what does not and why."""
    parts = []
    if contacts is not None:
        t, w = contacts['totals'], contacts['why']
        n_dis = t['fp'] + t['fn']
        top = max(WHY, key=lambda k: w[k]) if n_dis else None
        parts.append(f'Contact points: the render agrees with the grasp study on {100 * t["agreement"]:.1f} % of '
                     f'{t["n"]:,} points in view over {contacts["cases"]} poses'
                     + (f' ({t["fp"]:,} predicted visible but hidden in the render, {t["fn"]:,} the other way; most '
                        f'{WHY_TEXT[top].split(" (")[0]})' if n_dis else '') + '.')
        g = contacts['grasp_level']
        if g['n']:
            parts.append(f'The grasp study\'s requirement gives the same verdict on {g["tp"] + g["tn"]} of {g["n"]} '
                         'grasps.')
    if depth is not None:
        t = depth['totals']
        on = {x.get('compared_on') for x in depth.get('sensors') or []}
        what = ('task-volume voxels' if on == {'voxels'} else 'rendered surface pixels' if on == {'surfaces'}
                else 'task-volume voxels and rendered surface pixels')
        parts.append(f'Depth validity: {100 * t["agreement"]:.1f} % of {t["n"]:,} {what} agree '
                     f'(IoU of the valid sets {100 * (t["iou"] or 0):.1f} %).')
        surf = [(x['name'], x['surfaces']) for x in depth.get('sensors') or []
                if x.get('compared_on') == 'voxels' and (x.get('surfaces') or {}).get('agreement') is not None]
        if surf:
            name, sf = min(surf, key=lambda z: z[1]['agreement'])
            inside = (sf.get('inside_volumes') or {}).get('agreement')
            parts.append(f'On the surfaces the imagers see, the lowest is {name}: {100 * sf["agreement"]:.1f} % of '
                         f'{sf["n"]:,} pixels'
                         + (f' ({100 * inside:.1f} % inside the task volume)' if inside is not None else '') + '.')
    if s['status'] != 'good':
        parts.append('Look at the worst disagreements below: an occlusion disagreement points at the prediction\'s '
                     'geometry (the cube-map size, the camera\'s seat), a definition one at the visibility rule.')
    return ' '.join(parts)


def _enc(a):
    from .coverage import encode_array
    return encode_array(a)


def decode(doc: dict) -> dict:
    """The contact comparison's arrays: 'states' (cases, cameras, points) uint8 (ST_*), 'points' (cases, points, 3)
    world metres, 'fingertip' (points,) index into contacts['fingertips']; {} when there is none."""
    from .coverage import decode_array
    c = (doc or {}).get('contacts') or {}
    if 'states' not in c:
        return {}
    return {'states': decode_array(c['states']), 'points': decode_array(c['points']).astype(float),
            'fingertip': decode_array(c['point_fingertip']).astype(int)}


def decode_thumb(t: dict) -> dict:
    from .coverage import decode_array
    return {'predicted_rgb': decode_array(t['predicted_rgb']), 'rendered_rgb': decode_array(t['rendered_rgb']),
            'uv': decode_array(t['uv']).astype(float), 'pred': decode_array(t['pred']).astype(bool),
            'rend': decode_array(t['rend']).astype(bool)}
