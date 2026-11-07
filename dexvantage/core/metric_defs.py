"""The named figures of merit of every study (core/metrics.py): each study's headline figure and its obvious secondary
figures, read off the study's own result. The tolerance Monte Carlo, the designed experiments, the optimiser, the
design agent, the requirements scorecard (core/requirements.py, batch --check) and the Python API all name figures by
these ids.

Every extractor reads a number the study already computed; nothing here models anything. A figure the result does not
state reads None ('not stated') with the reason when the study gives one; a figure the study states only as a bound
(the time-sync skew of a stream whose clock figures are not all stated is "at least"; the motion-blur smear over an
exposure that is not stated is "at most") carries bound 'at_least' or 'at_most', which the scorecard uses to pass or
fail only when the bound decides it.

Ids are '<area>.<figure>'; units are the result's own (percentages 0-100, millimetres, milliseconds, pixels, Mbit/s,
counts). The study ids are those of core/jobs.STUDIES.

Known limits: a metric is only what its study means by the figure (contact visibility is the grasp study's "every
fingertip at least 50 % visible in at least one camera"; coverage is the share of task-volume voxels seen, not of an
object's surface; the bandwidth margin is the specification's capacity bound minus the computed load, and so on: each
study's own Known limits apply). Counts of problems (cables flagged, controllers over budget) are the study's flags,
which are only as complete as the figures the library states. rig.cost adds up the cited "price" blocks of the
library entries (core/prices.py): a lower bound ('at_least', the parts without a price counted in the reason) when
some parts state none, not stated when none does, and refused when the prices are not all in US dollars (no exchange
rate is assumed).
"""
from __future__ import annotations

import math

from .metrics import metric


def _pct(x):
    return None if x is None else float(x) * 100.0


def _get(doc, *path):
    cur = doc
    for k in path:
        if cur is None:
            return None
        cur = cur.get(k) if isinstance(cur, dict) else (cur[k] if isinstance(cur, list) and -len(cur) <= k < len(cur)
                                                        else None)
    return cur


def _why(value, reason: str):
    """value, or None with the plain reason it is not stated."""
    return value if value is not None else {'value': None, 'reason': reason}


# ====================================================================== grasp self-occlusion (§7.3)
@metric('grasp.contact_visibility', 'Contact visibility', '%', 'higher', study='grasp_visibility',
        note='share of the reachable grasps with every fingertip at least 50 % visible in at least one camera',
        tags=('headline',))
def _grasp_cvs(doc):
    return _get(doc, 'summary', 'cvs_pct')


@metric('grasp.contact_visibility_low', 'Contact visibility, lower bound of its interval', '%', 'higher',
        study='grasp_visibility', note='the lower end of the grasp study\'s bootstrap interval (its ci_level)')
def _grasp_cvs_low(doc):
    return _pct(_get(doc, 'summary', 'ci_low'))


@metric('grasp.both_fingertips', 'Both fingertips seen', '%', 'higher', study='grasp_visibility',
        note='share of reachable grasps with every fingertip seen by one and the same camera')
def _grasp_both(doc):
    return _pct(_get(doc, 'summary', 'both_fingertips_rate'))


@metric('grasp.reachable', 'Grasps reachable', '%', 'higher', study='grasp_visibility',
        note='feasible (IK, joint limits, no collision) share of the grasp set')
def _grasp_reach(doc):
    s = doc.get('summary') or {}
    n, f = s.get('grasps'), s.get('feasible')
    return None if not n or f is None else 100.0 * f / n


# ====================================================================== coverage (§7.1)
@metric('coverage.covered', 'Task volume seen', '%', 'higher', study='coverage', tags=('headline',),
        note='share of the task-volume voxels seen by at least one camera')
def _cov_seen(doc):
    return _pct(_get(doc, 'summary', 'seen_ge1'))


@metric('coverage.seen_twice', 'Task volume seen by two or more cameras', '%', 'higher', study='coverage')
def _cov_two(doc):
    return _pct(_get(doc, 'summary', 'seen_ge2'))


@metric('coverage.blind', 'Task volume blind', '%', 'lower', study='coverage')
def _cov_blind(doc):
    return _pct(_get(doc, 'summary', 'blind'))


@metric('coverage.ppmm_p5', 'Resolution, 5th percentile', 'px/mm', 'higher', study='coverage',
        note='best camera\'s pixels per millimetre, 5th percentile over the seen voxels')
def _cov_ppmm(doc):
    return _get(doc, 'summary', 'best_ppmm_p5')


# ====================================================================== depth validity (§7.2)
@metric('depth.valid', 'Depth-valid task volume', '%', 'higher', study='depth_validity', tags=('headline',),
        note='share of the task volume where some depth sensor gives valid depth')
def _depth_valid(doc):
    return _why(_get(doc, 'rig', 'valid_pct'), doc.get('message') or 'The rig has no depth sensor to evaluate.')


@metric('depth.worst_sensor_valid', 'Depth-valid share of the weakest depth sensor', '%', 'higher',
        study='depth_validity')
def _depth_worst(doc):
    vals = [s.get('valid_pct') for s in doc.get('sensors') or [] if s.get('valid_pct') is not None]
    return _why(min(vals) if vals else None, 'The rig has no depth sensor to evaluate.')


@metric('depth.sigma_z_p95_mm', 'Depth noise, 95th percentile (worst sensor)', 'mm', 'lower', study='depth_validity',
        note='1σ depth error of the stereo model, 95th percentile over each sensor\'s valid voxels; the worst sensor')
def _depth_sigma(doc):
    vals = [_get(s, 'sigma_z', 'p95_m') for s in doc.get('sensors') or []]
    vals = [v for v in vals if v is not None]
    return max(vals) * 1000.0 if vals else None


# ====================================================================== bandwidth, hosts and power (§7.6)
def _controllers(doc):
    return [c for c in doc.get('controllers') or [] if isinstance(c, dict)]


@metric('bandwidth.margin', 'Smallest controller margin', 'Mbit/s', 'higher', study='bandwidth', tags=('headline',),
        note='capacity minus load of the busiest-relative controller (the specification bound unless measured)')
def _bw_margin(doc):
    cs = _controllers(doc)
    known = [c['margin_mbps'] for c in cs if c.get('margin_mbps') is not None]
    if not cs:
        return {'value': None, 'reason': 'The rig has no host controller: add the host on the Bandwidth page.'}
    if not known:
        return {'value': None, 'reason': 'No controller has a stated capacity.'}
    out = {'value': min(known)}
    if len(known) < len(cs) or any(c.get('incomplete') for c in cs):
        out['note'] = 'some controllers or streams have no stated figure: the margin of the others'
    return out


@metric('bandwidth.busiest_load', 'Busiest link or controller load', '%', 'lower', study='bandwidth')
def _bw_busiest(doc):
    return _why(_get(doc, 'headline', 'busiest_load_pct'), 'No link or controller has a stated capacity and load.')


@metric('bandwidth.over_budget', 'Controllers over budget', 'count', 'lower', study='bandwidth',
        note='controllers whose load exceeds their capacity')
def _bw_over(doc):
    cs = _controllers(doc)
    if not cs:
        return {'value': None, 'reason': 'The rig has no host controller: add the host on the Bandwidth page.'}
    n = sum(1 for c in cs if c.get('load') is not None and c['load'] > 1.0 + 1e-12)
    unknown = sum(1 for c in cs if c.get('load') is None) or any(c.get('incomplete') for c in cs) or \
        bool(_get(doc, 'headline', 'incomplete_sensors'))
    if unknown:
        return {'value': n, 'bound': 'at_least',
                'note': 'a controller or stream without a stated figure is not counted: at least this many'}
    return n


@metric('bandwidth.payload_mbps', 'Total payload', 'Mbit/s', 'lower', study='bandwidth')
def _bw_payload(doc):
    v = _get(doc, 'headline', 'total_payload_mbps')
    if v is not None and _get(doc, 'headline', 'incomplete_sensors'):
        return {'value': v, 'bound': 'at_least', 'note': 'streams without a stated rate are not counted'}
    return v


# ====================================================================== time sync (§7.10)
def _sync_bound(doc, value):
    s = doc.get('summary') or {}
    if value is not None and s.get('complete') is False:
        return {'value': value, 'bound': 'at_least',
                'note': 'some clock or exposure figures are not stated: the skew is at least this'}
    return value


@metric('sync.skew_ms', 'Worst camera to joint-state skew', 'ms', 'lower', study='time_sync', tags=('headline',))
def _sync_skew(doc):
    v = _get(doc, 'summary', 'worst_skew_s')
    return _sync_bound(doc, None if v is None else v * 1000.0)


@metric('sync.label_error_mm', 'Worst action-label error from the skew', 'mm', 'lower', study='time_sync',
        note='tool-centre-point travel during the skew along the trajectories')
def _sync_label(doc):
    v = _get(doc, 'summary', 'label_error_m')
    return _sync_bound(doc, None if v is None else v * 1000.0)


# ====================================================================== cables (§7.5)
def _cable_flags(doc):
    return _get(doc, 'summary', 'flags') or {}


def _cable_unknown(doc, n):
    if _cable_flags(doc).get('unknown'):
        return {'value': n, 'bound': 'at_least',
                'note': 'some cable figures are not stated (length, bend radius): at least this many'}
    return n


@metric('cables.worst', 'Cables not clear', 'count', 'lower', study='cables', tags=('headline',),
        note='cables with any pull, snag, bend or unchecked figure')
def _cab_worst(doc):
    s = doc.get('summary') or {}
    if s.get('cables') is None:
        return None
    return int(s['cables']) - int(s.get('clear') or 0)


@metric('cables.pull_snag', 'Cable pulls and snags', 'count', 'lower', study='cables',
        note='pull and snag flags over every cable, run and checked pose')
def _cab_ps(doc):
    f = _cable_flags(doc)
    if (doc.get('summary') or {}).get('cables') is None:
        return None
    return _cable_unknown(doc, int(f.get('pull', 0)) + int(f.get('snag', 0)))


@metric('cables.bend', 'Cable bends under the allowed radius', 'count', 'lower', study='cables')
def _cab_bend(doc):
    if (doc.get('summary') or {}).get('cables') is None:
        return None
    return _cable_unknown(doc, int(_cable_flags(doc).get('bend', 0)))


# ====================================================================== VLA tokens (§7.12)
@metric('vla.min_tokens', 'Fewest ViT tokens on a critical object', 'tokens', 'higher', study='vla_tokens',
        tags=('headline',))
def _vla_min(doc):
    return _why(_get(doc, 'summary', 'fewest', 'tokens'),
                'No critical task object is in a policy input to count (mark the object as critical on the Workspace '
                'page).')


@metric('vla.fingertip_tokens', 'Fewest ViT tokens on a fingertip', 'tokens', 'higher', study='vla_tokens')
def _vla_tips(doc):
    return _get(doc, 'summary', 'fingertips_fewest', 'tokens')


@metric('vla.warnings', 'Token warnings', 'count', 'lower', study='vla_tokens')
def _vla_warn(doc):
    return _get(doc, 'summary', 'warnings')


# ====================================================================== action chunk (§7.13)
def _chunk(doc, key):
    v = _get(doc, 'headline', key)
    if v is None:
        return None
    if any(r.get('missing') for r in doc.get('policies') or [] if isinstance(r, dict)):
        return {'value': v, 'note': 'some loop figures are not stated: the loop is "at least" without them'}
    return v


@metric('action_chunk.air_grasp_mm', 'Worst air grasp', 'mm', 'lower', study='action_chunk_sync',
        note='distance short of contact at which the delayed chunk closes the gripper')
def _ac_air(doc):
    return _chunk(doc, 'air_grasp_mm')


@metric('action_chunk.worst_mm', 'Worst closing error', 'mm', 'lower', study='action_chunk_sync', tags=('headline',))
def _ac_worst(doc):
    return _chunk(doc, 'worst_mm')


@metric('action_chunk.overshoot_mm', 'Worst overshoot', 'mm', 'lower', study='action_chunk_sync')
def _ac_over(doc):
    return _chunk(doc, 'overshoot_mm')


@metric('action_chunk.reaction_mm', 'Reaction distance', 'mm', 'lower', study='action_chunk_sync')
def _ac_react(doc):
    return _chunk(doc, 'reaction_mm')


# ====================================================================== teleop (§7.4)
@metric('teleop.overlap', 'Teleoperable task volume', '%', 'higher', study='teleop', tags=('headline',))
def _teleop(doc):
    return _why(_get(doc, 'summary', 'teleoperable_pct'),
                'There is no teleop device with stated figures: add one on the Teleop overlap page.')


# ====================================================================== lighting (§7.8)
@metric('lighting.glare', 'Specular saturation (worst camera)', '%', 'lower', study='lighting', tags=('headline',))
def _light_glare(doc):
    h = doc.get('headline') or {}
    v = h.get('value')
    if v is None:
        return {'value': None, 'reason': h.get('text') or 'No camera sees the object or the gripper.'}
    return {'value': v, 'note': 'relative model (no light power stated)'} if h.get('relative') else v


@metric('lighting.depth_loss', 'Depth lost to lighting (worst sensor)', '%', 'lower', study='lighting')
def _light_depth(doc):
    h = doc.get('headline') or {}
    return _why(h.get('depth_loss_pct'), h.get('text') if h.get('value') is None else
                'No depth sensor sees the object or the gripper.')


# ====================================================================== motion blur (§7.9)
@metric('motion_blur.worst_px', 'Worst motion smear', 'px', 'lower', study='motion_blur', tags=('headline',))
def _mb_smear(doc):
    s = doc.get('summary') or {}
    v = s.get('smear_px')
    if v is None:
        return {'value': None, 'reason': s.get('advice') or f'The motion-blur study says "{s.get("word")}".'}
    if s.get('smear_bound'):
        return {'value': v, 'bound': 'at_most', 'note': 'exposure from the maximum the stream allows: at most this'}
    return v


@metric('motion_blur.skew_px', 'Worst rolling-shutter skew', 'px', 'lower', study='motion_blur')
def _mb_skew(doc):
    s = doc.get('summary') or {}
    v = s.get('skew_px')
    if v is not None and s.get('skew_bound'):
        return {'value': v, 'bound': 'at_most', 'note': 'readout time not stated: at most this'}
    return v


# ====================================================================== hand-eye plan (§8.1)
def _he(doc, keys):
    sx = _get(doc, 'sigma', 'X') or {}
    vals = [sx.get(k) for k in keys]
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    if not vals:
        return {'value': None, 'reason': doc.get('advice') or 'The plan does not determine the camera mount.'}
    return max(vals)


@metric('handeye.sigma_mm', 'Predicted camera position σ (worst axis)', 'mm', 'lower', study='handeye_plan',
        tags=('headline',), note='1σ of the camera mount (X) from the planned poses, largest of x, y, z')
def _he_mm(doc):
    return _he(doc, ('x_mm', 'y_mm', 'z_mm'))


@metric('handeye.sigma_deg', 'Predicted camera rotation σ (worst axis)', '°', 'lower', study='handeye_plan')
def _he_deg(doc):
    return _he(doc, ('rx_deg', 'ry_deg', 'rz_deg'))


@metric('handeye.ratio', 'Predicted σ over the requirement (worst)', 'ratio', 'lower', study='handeye_plan',
        note='1 or less meets the plan\'s own requirement')
def _he_ratio(doc):
    return _get(doc, 'worst', 'ratio')


# ====================================================================== MuJoCo ground truth (§9.1)
@metric('groundtruth.agreement', 'Rendered against predicted agreement', '%', 'higher', study='groundtruth',
        tags=('headline',))
def _gt(doc):
    return _get(doc, 'summary', 'agreement_pct')


@metric('groundtruth.depth_agreement', 'Rendered depth agreement', '%', 'higher', study='groundtruth')
def _gt_depth(doc):
    return _get(doc, 'summary', 'depth_agreement_pct')


# ====================================================================== multi-finger hand (§7.14, §7.15)
@metric('dexhand.fingertips_seen', 'Fingertips seen', '%', 'higher', study='dex_occlusion', tags=('headline',))
def _dex_seen(doc):
    return _get(doc, 'rig', 'fingertips_seen_pct')


@metric('dexhand.all_fingertips', 'Grasps with every fingertip seen', '%', 'higher', study='dex_occlusion')
def _dex_all(doc):
    return _get(doc, 'rig', 'all_fingertips_rate_pct')


def _handoff(doc):
    h = doc.get('handoff')
    return h if isinstance(h, dict) else None


@metric('dexhand.blind_gap_mm', 'Sensory blind gap before contact (worst)', 'mm', 'lower', study='dex_occlusion',
        note='approach distance over which a fingertip is neither seen nor yet touching')
def _dex_gap(doc):
    h = _handoff(doc)
    if h is None or h.get('blind_gap_m') is None:
        return {'value': None, 'reason': 'The study did not run the visual-to-tactile handoff.'}
    v = h['blind_gap_m'] * 1000.0
    censored = any(r.get('censored') for a in h.get('approach_visibility') or [] for r in a.get('fingers') or []
                   if isinstance(r, dict) and r.get('blind_gap_m') == h['blind_gap_m'])
    return {'value': v, 'bound': 'at_least', 'note': 'blind from the start of the scan: at least this'} if censored \
        else v


@metric('dexhand.unsensed', 'Fingertips neither seen nor touched at contact', 'count', 'lower',
        study='dex_occlusion')
def _dex_unsensed(doc):
    h = _handoff(doc)
    if h is None:
        return {'value': None, 'reason': 'The study did not run the visual-to-tactile handoff.'}
    return sum(1 for a in h.get('approach_visibility') or [] for r in a.get('fingers') or []
               if isinstance(r, dict) and r.get('sensed_at_contact') is False)


# ====================================================================== brackets (§7.7)
@metric('brackets.not_ready', 'Sensors without a ready bracket', 'count', 'lower', study='brackets',
        tags=('headline',))
def _br(doc):
    r = doc.get('rig') or {}
    if r.get('brackets') is None:
        return None
    return int(r['brackets']) - int(r.get('ready') or 0) + int(r.get('unset') or 0)


# ====================================================================== canonical view warp (§7.16)
@metric('canonical_view.parallax_p95', 'Canonical view warp error off the contact plane (P95)', 'px', 'lower',
        study='canonical_view', tags=('headline',),
        note='the residual parallax at the worse end of the depth range: the larger of its 95th percentiles at the '
             'two ends, over the canonical pixels the warp fills (the study\'s headline)')
def _canon_p95(doc):
    return _why(_get(doc, 'headline', 'value'), _get(doc, 'headline', 'advice') or 'no pixel of the canonical view '
                'can be filled')


@metric('canonical_view.coverage', 'Canonical image the warp fills', '%', 'higher', study='canonical_view')
def _canon_cover(doc):
    return _pct(_get(doc, 'coverage', 'share'))


# ====================================================================== data (§8.2-§8.4)
@metric('audit.pass_rate', 'Episodes that pass the audit', '%', 'higher', study='audit', tags=('headline',))
def _audit(doc):
    return _get(doc, 'summary', 'pass_pct')


@metric('plan.match', 'Plan against recording: comparisons that match', '%', 'higher', study='plan_vs_recording',
        tags=('headline',))
def _pvr(doc):
    return _get(doc, 'summary', 'match_pct')


@metric('silhouette.offset_mm', 'Mount offset found by the silhouettes', 'mm', 'lower', study='silhouette_calib',
        tags=('headline',))
def _sil_mm(doc):
    return _get(doc, 'headline', 'offset_mm')


@metric('silhouette.offset_deg', 'Mount rotation found by the silhouettes', '°', 'lower', study='silhouette_calib')
def _sil_deg(doc):
    return _get(doc, 'headline', 'offset_deg')


# ====================================================================== the rig itself (core/studies/rig_inventory.py)
@metric('rig.camera_count', 'Cameras', 'count', 'lower', study='rig', tags=('headline',),
        note='enabled camera and depth sensors of the variant')
def _rig_cams(doc):
    return doc.get('cameras')


@metric('rig.sensor_count', 'Sensors', 'count', 'lower', study='rig')
def _rig_sensors(doc):
    return doc.get('sensors')


def _rig_total(doc, key):
    t = doc.get(key) or {}
    if t.get('value') is None:
        return {'value': None, 'reason': t.get('reason') or 'not stated by the maker'}
    return t['value']


@metric('rig.cost', 'Rig cost (where the library states a price)', 'USD', 'lower', study='rig',
        note='sum of the cited list prices; a lower bound when some parts state no price (they are counted)')
def _rig_cost(doc):
    t = doc.get('cost') or {}
    if t.get('value') is None:
        return {'value': None, 'reason': t.get('reason') or 'A price is not stated.'}
    if t.get('unit') and t['unit'] != 'USD':          # the metric is in USD; no exchange rate is assumed
        return {'value': None, 'reason': f'The prices are in {t["unit"]}, and rig.cost is in US dollars: no exchange '
                                         'rate is assumed, so it is not stated. The rig inventory gives the total in '
                                         f'{t["unit"]}.'}
    if t.get('bound'):
        return {'value': t['value'], 'bound': t['bound'], 'reason': t.get('reason') or ''}
    return t['value']


@metric('rig.mass_g', 'Sensor mass', 'g', 'lower', study='rig')
def _rig_mass(doc):
    return _rig_total(doc, 'mass_g')


@metric('rig.power_w', 'Sensor power', 'W', 'lower', study='rig')
def _rig_power(doc):
    return _rig_total(doc, 'power_w')


# ====================================================================== multi-cell comparison (core/cells.py)
@metric('cells.gap_max', 'Largest geometric gap to the reference cell', 'index', 'lower', study='cells',
        tags=('headline',), note='the gap index (0 identical … 1 unrelated, a heuristic) of the least alike cell')
def _cells_gap(doc):
    return _get(doc, 'headline', 'value')


@metric('cells.not_poolable', 'Cells whose recordings cannot be pooled', 'count', 'lower', study='cells',
        note='cells with a feature no mapping step reaches (another joint count, a missing camera view)')
def _cells_nopool(doc):
    return _get(doc, 'pooling', 'counts', 'not poolable')


@metric('cells.pool_steps', 'Mapping steps to pool the recordings', 'count', 'lower', study='cells',
        note='rename, resize, resample, reorder, re-encode and rescale steps over every cell')
def _cells_steps(doc):
    return _get(doc, 'pooling', 'operations')


@metric('cells.drift_worst', 'Worst measured drift of a built copy', 'ratio', 'lower', study='cells',
        note='the largest share of the mount tolerance a calibrated camera of a copy sits off the plan')
def _cells_drift(doc):
    return _why(_get(doc, 'drift', 'worst', 'ratio'), 'No cell has a hand-eye calibration of a camera the plan has.')


@metric('cells.meaningful_differences', 'Figures that differ beyond the tolerance spread', 'count', 'lower',
        study='cells', note='differences from the reference larger than 2 σ of the tolerance Monte Carlo')
def _cells_meaningful(doc):
    m = doc.get('metrics') or {}
    if not _get(m, 'tolerance', 'draws'):
        return {'value': None, 'reason': 'No tolerance draws were made, so no difference was judged.'}
    return m.get('meaningful')


# ====================================================================== the experiment tracker (core/tracking)
@metric('tracking.runs', 'Runs logged', 'count', 'higher', study='tracking', tags=('headline',),
        note='runs in the project\'s run store (every study run, experiment, optimiser run, tolerance Monte Carlo '
             'and scorecard logged while tracking is on)')
def _tracking_runs(doc):
    return _get(doc, 'counts', 'runs')


# ====================================================================== data-collection campaign (core/campaign.py)
def _campaign_storage(doc):
    t = doc.get('totals') or {}
    v = t.get('storage_gb')
    if v is None:
        return None
    if t.get('storage_bound') == 'at_least':
        return {'value': v, 'bound': 'at_least', 'note': 'a sensor\'s data rate is not stated: '
                + ', '.join(t.get('storage_incomplete') or [])}
    return v


@metric('campaign.episodes', 'Campaign episodes recorded', 'count', 'higher', study='campaign', tags=('headline',),
        note='every episode the plan schedules (all conditions and operators)')
def _campaign_episodes(doc):
    return _get(doc, 'totals', 'episodes')


@metric('campaign.days', 'Campaign days', 'days', 'lower', study='campaign',
        note='recording days the schedule takes within the stated session and day limits')
def _campaign_days(doc):
    return _get(doc, 'totals', 'days')


@metric('campaign.storage_gb', 'Campaign storage', 'GB', 'lower', study='campaign',
        note='the bandwidth study\'s data rates × the recording format\'s ratio × the container overhead × the '
             'recorded time; at least, when a sensor\'s rate is not stated')
def _campaign_gb(doc):
    return _campaign_storage(doc)


@metric('campaign.rig_hours', 'Campaign hours on the rig', 'h', 'lower', study='campaign',
        note='the sessions\' length: episodes, resets, changeovers and per-session calibration checks')
def _campaign_hours(doc):
    return _get(doc, 'totals', 'rig_hours')


@metric('campaign.kept_per_condition', 'Expected kept episodes per condition (fewest)', 'count', 'higher',
        study='campaign', note='episodes recorded per condition × the keep rate (a default of 1 until set)')
def _campaign_kept(doc):
    cs = doc.get('conditions') or []
    return min((c.get('expected_kept') for c in cs if c.get('expected_kept') is not None), default=None)


# ====================================================================== the sim-to-real gap (core/sim2real/gap.py)
@metric('sim2real.image_gap_dn', 'Sim-to-real image gap (worst camera)', 'DN', 'lower', study='sim2real',
        tags=('headline',), note='mean absolute difference between the recorded and the rendered frames, grey levels, '
                                 'before any correction')
def _s2r_gap(doc):
    return _get(doc, 'summary', 'gap_dn')


@metric('sim2real.remaining_dn', 'Sim-to-real gap left after the corrections (worst camera)', 'DN', 'lower',
        study='sim2real', note='the image difference with every correction found applied')
def _s2r_left(doc):
    return _get(doc, 'summary', 'remaining_dn')


@metric('sim2real.explained', 'Sim-to-real gap explained by the corrections (worst camera)', '%', 'higher',
        study='sim2real')
def _s2r_explained(doc):
    return _get(doc, 'summary', 'explained_pct')


def _s2r_worst(doc, fn, reason):
    vals = [v for v in (fn(c) for c in doc.get('cameras') or []) if v is not None]
    return max(vals) if vals else {'value': None, 'reason': reason}


@metric('sim2real.latency_ms', 'Camera latency against the joint states (largest)', 'ms', 'lower', study='sim2real',
        note='|τ|: the time the frames show the robot later (or earlier) than their stamps, from render-and-compare')
def _s2r_latency(doc):
    return _s2r_worst(doc, lambda c: abs(c['latency']['latency_s']) * 1000 if (c.get('latency') or {}).get(
        'observable') else None, 'no camera\'s latency is observable')


@metric('sim2real.mount_offset_mm', 'Mount offset found by the silhouettes (largest)', 'mm', 'lower',
        study='sim2real')
def _s2r_mount(doc):
    return _s2r_worst(doc, lambda c: (c.get('mount') or {}).get('delta', {}).get('norm_mm') if (
        c.get('mount') or {}).get('observable') else None, 'no static camera\'s mount is observable')


@metric('sim2real.silhouette_iou', 'Robot silhouette IoU after the corrections (worst static camera)', '%', 'higher',
        study='sim2real')
def _s2r_iou(doc):
    vals = [c['silhouette']['iou_after'] for c in doc.get('cameras') or []
            if (c.get('silhouette') or {}).get('iou_after') is not None]
    return 100.0 * min(vals) if vals else {'value': None, 'reason': 'no static camera to segment the robot in'}


# ====================================================================== statistics and power (core/studies/stats.py)
@metric('stats.p_adjusted', 'Smallest corrected p-value of the comparisons', 'p', 'lower', study='stats',
        tags=('headline',), note='the headline comparison\'s p-value after the correction across the comparisons')
def _stats_p(doc):
    h = doc.get('headline') or {}
    return _why(h.get('p_adjusted'), 'the statistics result has no comparison (only the power calculator)')


@metric('stats.n_needed', 'Episodes or units needed per condition (power calculator)', 'count', 'lower',
        study='stats', note='the power calculator\'s n per condition for the target power')
def _stats_n(doc):
    return _why(_get(doc, 'power', 'n_per_group'), 'the power calculator is not filled in')


@metric('stats.planned_power', 'Power of the planned episodes per condition', '%', 'higher', study='stats',
        note='the power calculator\'s power at the planned n')
def _stats_planned(doc):
    return _why(_pct(_get(doc, 'power', 'planned', 'power')), 'no planned number of episodes is set')
