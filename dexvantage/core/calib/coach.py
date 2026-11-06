"""Hand-eye calibration coach: given the poses taken so far, which pose to take next and why.

Ported from Vantage core/calib_coach.py (the next-best-view idea): the information of the views already accepted
adds up; every candidate the operator could take next is scored by what its information would add; a candidate that
duplicates a view already taken (the camera within DUP_M of it and turned less than DUP_DEG, Vantage's rule) is left
out; the advice says which axis is the loosest now, what the chosen pose does to it and how to get there. What
changed: the views are robot poses of a hand-eye calibration (core/calib/planner.py's candidates and Fisher
information: X and Z of the bundle adjustment, one camera, one board), not a board held by hand in front of a lens.

The information of the poses taken: each pose's detected corner ids when it has them (a capture session), else the
corners the planner's rules count at that pose (a planned pose), at the corner noise σ (the solve's residual σ̂ when
there is a solve, else the planner's PIXEL_SIGMA default). The linearisation point is the solve's X and Z when there
is one, else the nominal mounts.

Criterion (PlanSettings.criterion): 'weakest' (the default, Vantage's coach default) takes the candidate that leaves
the smallest worst-axis ratio of the camera (its largest σ over the requirement), so the operator firms up the loosest
axis first; while the poses taken do not yet pin every axis (and between equal candidates) it takes the largest
log det of the information, the largest reduction of the covariance's volume ('d_optimal', which can also be chosen
outright). The chosen candidate is checked for occlusion before it is advised.

'enough': every axis of the camera is within the requirement; the advice then says so, and what one more pose would
still gain.

Known limits: those of planner.py (exact joint readings, known intrinsics, independent corner noise, a fixed grid of
candidates); the advice moves the robot to the candidate's joint vector, not along a collision-checked path between
poses (check the motion on the robot, or step it through the Camera view).
"""
from __future__ import annotations

import math

import numpy as np

from . import planner as PL

DUP_M = PL.DUP_M
DUP_DEG = PL.DUP_DEG


def taken_views(rig: PL.Rig, poses: list, st: PL.PlanSettings) -> list:
    """[(q, ids)] of pose records ({'q', 'ids'}): detected ids when given, else the corners counted at q."""
    out = []
    nq = rig.model.nq
    for p in poses:
        q = np.asarray(p.get('q') or [], float)
        if len(q) != nq:
            continue
        ids = np.asarray(p.get('ids') or [], int)
        if not len(ids):
            ids, _why = PL.counted(rig, q, st)
        if len(ids):
            out.append((q, ids))
    return out


def _duplicates(rig: PL.Rig, cs: PL.Candidates, views: list) -> np.ndarray:
    """(C,) True for candidates that repeat a view taken (camera within DUP_M and its axis within DUP_DEG)."""
    dup = np.zeros(len(cs), bool)
    if not views or not len(cs):
        return dup
    taken = [rig.camera_world(q) for q, _ids in views]
    P = np.array([T[:3, 3] for T in taken])
    Zs = np.array([T[:3, 2] for T in taken])
    Rs = np.array([T[:3, :3] for T in taken])
    cosd = math.cos(math.radians(DUP_DEG))
    for c in range(len(cs)):
        T = rig.camera_world(cs.q[c])
        near = np.linalg.norm(P - T[:3, 3], axis=1) <= DUP_M
        if not near.any():
            continue
        same_axis = Zs @ T[:3, 2] >= cosd
        # the turn between the two frames, roll included
        tr = np.einsum('kij,ij->k', Rs, T[:3, :3])
        same_turn = (tr - 1) / 2 >= cosd
        dup[c] = bool((near & same_axis & same_turn).any())
    return dup


def advise(rig: PL.Rig, poses: list, st: PL.PlanSettings | None = None, cs: PL.Candidates | None = None,
           q_now=None, sigma_px: float | None = None) -> dict:
    """The next pose (module docstring). poses: the pose records taken so far; q_now: where the robot is (default the
    last pose taken, else rig.q_hold); sigma_px overrides the corner noise. JSON-able."""
    st = st or PL.PlanSettings()
    if sigma_px is not None and math.isfinite(sigma_px) and sigma_px > 0:
        st = PL.PlanSettings.from_dict(dict(st.to_dict(), pixel_sigma=float(sigma_px)))
    views = taken_views(rig, poses, st)
    H = PL.information(rig, views, st.pixel_sigma)
    sx, sz, _C, det = PL.sigmas(H)
    ratio, key = PL.worst(sx, st.need_mm, st.need_deg)
    sev, word = PL.verdict(ratio)
    cs = cs if cs is not None else PL.candidates(rig, st)
    dup = _duplicates(rig, cs, views)
    avail = ~dup
    c = PL.next_best(rig, cs, H, st, avail)
    out = {'taken': len(views), 'sigma_X': PL._sig_doc(sx), 'sigma_Z': PL._sig_doc(sz), 'determined': bool(det),
           'worst': {'ratio': PL._r(ratio), 'key': key, 'axis': PL.AXIS_WORDS.get(key.split('_')[0], key) if key
                     else '', 'value': PL._r(sx.get(key, math.inf), 5)},
           'status': sev, 'word': word, 'enough': bool(math.isfinite(ratio) and ratio <= 1.0),
           'criterion': st.criterion, 'pixel_sigma': st.pixel_sigma,
           'candidates': {'total': cs.total, 'feasible': len(cs), 'duplicates': int(dup.sum())},
           'requirement': {'need_mm': st.need_mm, 'need_deg': st.need_deg}}
    if c is None:
        out['next'] = None
        out['advice'] = ('No candidate pose is left that the robot reaches with the board in view: the poses taken '
                         'cover the plan\'s candidates.' if len(cs) else
                         'No candidate pose is reachable with the board in view: move the board or the camera.')
        return out
    g = PL.gain_doc(rig, cs, H, c, st)
    g['what'] = PL.describe_view(cs.meta[c])
    q_from = np.asarray(q_now, float) if q_now is not None else (views[-1][0] if views else rig.q_hold)
    names = list(rig.model.joint_names)
    moves = []
    for j in np.flatnonzero(rig.chain):
        d = float(cs.q[c][j] - q_from[j])
        moves.append({'joint': names[j], 'from_deg': PL._r(math.degrees(q_from[j]), 2),
                      'to_deg': PL._r(math.degrees(cs.q[c][j]), 2), 'delta_deg': PL._r(math.degrees(d), 2)})
    g['moves'] = moves
    g['joint_names'] = names
    out['next'] = g
    out['advice'] = _why(out, g, st, len(views))
    return out


def _why(out: dict, g: dict, st: PL.PlanSettings, n: int) -> str:
    key = out['worst']['key']
    after_key = g.get('worst_axis_after') or key
    if n == 0:
        return f'Start with this pose: the camera {g["what"]}.'
    if not out['determined'] or out['worst']['ratio'] is None:
        return (f'The {n} pose{"s" if n != 1 else ""} taken do not pin every axis of the camera yet. Take this pose '
                f'next (the camera {g["what"]}): it adds the most information (log det +{g["logdet_gain"]:.2f}).')
    now = PL.fmt_sigma(key, out['sigma_X'][key])
    after = PL.fmt_sigma(key, (g['sigma_X_after'] or {}).get(key) or math.inf)
    pct = g.get('gain_pct', {}).get(key)
    head = (f'Enough: every axis of the camera is within {st.need_mm:g} mm and {st.need_deg:g}°. '
            if out['enough'] else '')
    s = (f'{head}The loosest axis now is the camera\'s {PL.AXIS_WORDS.get(key.split("_")[0], key)} at {now} '
         f'({out["worst"]["ratio"]:.2f}× the requirement). Next: the camera {g["what"]}; it brings that axis to '
         f'{after}' + (f' ({pct:.0f} % less)' if pct is not None else '') + '.')
    if after_key != key:
        s += (f' The loosest axis would then be its {PL.AXIS_WORDS.get(after_key.split("_")[0], after_key)}.')
    return s
