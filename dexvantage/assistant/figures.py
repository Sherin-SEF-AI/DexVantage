"""The figures a study's result yields, as the assistant's tools return them: a short, JSON-able brief read off the
study's own result document, never computed here and never filled in.

    brief(sid, doc) -> {'study', 'title', 'status', 'word', 'figure': {'value', 'unit', 'what'}, 'advice',
                        'suggestion', 'notes', 'defaults', 'rows', ..., 'seconds', 'cached'}
    compact(obj)    -> a small JSON-able copy (lists cut with their count, strings cut, encoded arrays, paths, URLs and
                       secrets left out, floats rounded to 4 significant decimals)

A figure the result does not state stays None. 'defaults' names the study settings that are DexVantage defaults
(the user's own figures are not), as each study records them. compact() is ported from Vantage
assistant/page_guide.py; the per-study briefs are DexVantage's.

Known limits: the brief picks the figures a reader needs first (the headline, the per-sensor rows, the suggestion);
run_study with full=true returns the whole result, compacted, for anything else.
"""
from __future__ import annotations

import math
import os
import re

_SECRET = re.compile(r'(api_?key|token|secret|password|passwd|credential|auth)', re.IGNORECASE)
_PATHISH = re.compile(r'(path|file|folder|dir|root|url|uri|command|cmd)s?$', re.IGNORECASE)
_DROP = ('schema', 'provenance', 'voxels', 'encoding', 'covariance', 'map', 'map_after', 'overlay', 'cube',
         'world_pose', 'grid', 'centers', 'link_world', 'camera_world', 'board_world')
_GONE = object()


def _looks_like_path(s: str) -> bool:
    s = s.strip()
    return bool(s) and (s.startswith(('/', '~', '\\\\')) or bool(re.match(r'^[A-Za-z]:[\\/]', s)) or
                        '://' in s or (os.sep in s and len(s) > 3 and ' ' not in s))


def short(text, n=240) -> str:
    text = ' '.join(str(text).split())
    if len(text) <= n:
        return text
    cut = text[:n - 1]
    if ' ' in cut[n // 2:]:
        cut = cut[:cut.rindex(' ')]
    return cut.rstrip(' ,;:-–') + '…'


def compact(o, depth=0, items=6, text=240, max_depth=4):
    out = _compact(o, depth, items, text, max_depth)
    return None if out is _GONE else out


def _compact(o, depth, items, text, max_depth):
    if isinstance(o, dict):
        if depth >= max_depth:
            return f'{{{len(o)} entries}}'
        out = {}
        for k, v in o.items():
            k = str(k)
            if k in _DROP or _SECRET.search(k) or _PATHISH.search(k):
                continue
            c = _compact(v, depth + 1, items, text, max_depth)
            if c is not _GONE:
                out[k] = c
        return out
    if isinstance(o, (list, tuple)):
        if depth >= max_depth:
            return f'[{len(o)} items]'
        vals = [_compact(v, depth + 1, items, text, max_depth) for v in list(o)[:items]]
        vals = [v for v in vals if v is not _GONE]
        return vals + ([f'… {len(o) - items} more'] if len(o) > items else [])
    if isinstance(o, bool) or o is None or isinstance(o, int):
        return o
    if isinstance(o, float):
        return _round(o)
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return _compact(o.item(), depth, items, text, max_depth)
        if isinstance(o, np.ndarray):
            return f'array {tuple(o.shape)}'
    except ImportError:                                  # pragma: no cover
        pass
    if isinstance(o, str):
        if _looks_like_path(o):
            return _GONE
        return short(o, text)
    return f'<{type(o).__name__}>'


def _round(v, nd=4):
    """v rounded for reading: nd decimals at 1 and above, nd significant digits below 1 (a skew of 0.000125 s stays
    0.000125, 47.30208 % reads 47.3021); None for what is not a finite number."""
    if v is None or isinstance(v, bool):
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    if v == 0 or abs(v) >= 1:
        return round(v, nd)
    return float(f'{v:.{nd}g}')


def _pct(frac):
    return None if frac is None else _round(100.0 * float(frac), 3)


def _get(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def _head(doc: dict, *blocks) -> dict:
    """status, word and advice from the first block that has them (headline, summary, rig, the document)."""
    out = {'status': None, 'word': None, 'advice': None}
    for b in [doc.get(k) for k in blocks] + [doc]:
        if not isinstance(b, dict):
            continue
        if out['status'] is None and isinstance(b.get('status'), str):
            out['status'] = b['status']
        if out['word'] is None:
            w = b.get('word') or b.get('status_word')
            if isinstance(w, str):
                out['word'] = w
        if out['advice'] is None and isinstance(b.get('advice'), str):
            out['advice'] = short(b['advice'], 600)
    return out


def _common(doc: dict) -> dict:
    st = doc.get('settings') if isinstance(doc.get('settings'), dict) else {}
    notes = [short(n, 300) for n in (doc.get('notes') or []) if isinstance(n, str)][:5]
    return {'suggestion': compact(doc.get('suggestion'), max_depth=2) if doc.get('suggestion') else None,
            'notes': notes, 'defaults': list(st.get('defaults') or [])}


def _fig(value, unit, what):
    return {'value': _round(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value,
            'unit': unit, 'what': what}


# =========================================================================== one brief per study
def _coverage(doc):
    s = doc.get('summary') or {}
    rows = [{'sensor': c.get('name'), 'sensor_uid': c.get('sensor_uid'), 'imager': c.get('imager'),
             'seen_pct': _pct(c.get('seen')), 'only_camera_pct': _pct(c.get('only_camera')),
             'hidden_pct': _pct(c.get('hidden')), 'outside_image_pct': _pct(c.get('outside_image')),
             'px_per_mm_median': _round(c.get('ppmm_median')), 'px_per_mm_p5': _round(c.get('ppmm_p5')),
             'range_median_m': _round(c.get('range_median_m')), 'on_link': c.get('on_link')}
            for c in doc.get('cameras') or []]
    return {**_head(doc, 'summary'),
            'figure': _fig(s.get('pct_ge2', _pct(s.get('seen_ge2'))), '%',
                           'task volume seen by two or more cameras'),
            'figures': {'seen_by_1_or_more_pct': _pct(s.get('seen_ge1')),
                        'seen_by_2_or_more_pct': _pct(s.get('seen_ge2')),
                        'blind_pct': _pct(s.get('blind')), 'resolved_by_1_or_more_pct': _pct(s.get('resolved_ge1')),
                        'best_px_per_mm_median': _round(s.get('best_ppmm_median')),
                        'best_px_per_mm_p5': _round(s.get('best_ppmm_p5')),
                        'redundancy_mean': _round(s.get('redundancy_mean')),
                        'voxels': _get(doc, 'grid', 'count'), 'voxel_mm': _get(doc, 'grid', 'voxel_mm')},
            'rows': rows, 'pose': compact(doc.get('q_source'))}


def _depth_validity(doc):
    rig = doc.get('rig') or {}
    rows = [{'sensor': s.get('name'), 'sensor_uid': s.get('uid'), 'model': s.get('title'),
             'technology': s.get('technology_title'), 'valid_pct': _round(s.get('valid_pct')),
             'status': s.get('status'), 'word': s.get('status_word'), 'min_z_m': _round(s.get('min_z_m')),
             'min_z_note': s.get('min_z_note'), 'max_range_m': _round(s.get('max_range_m')),
             'baseline_m': _round(s.get('baseline_m')), 'sigma_d_px': _round(s.get('sigma_d_px')),
             'sigma_d_default': s.get('sigma_d_default'), 'fails': compact(s.get('fails'), max_depth=1)}
            for s in doc.get('sensors') or []]
    return {**_head(doc, 'rig'), 'figure': _fig(rig.get('valid_pct'), '%', 'task volume depth-valid for the rig'),
            'rows': rows, 'skipped': compact(doc.get('skipped')), 'pose': compact(doc.get('pose'), max_depth=1),
            'message': doc.get('message') or None}


def _grasp_visibility(doc):
    s = doc.get('summary') or {}
    bm = doc.get('best_move') or {}
    return {**_head(doc, 'summary'),
            'figure': _fig(s.get('cvs_pct'), '%', 'reachable grasps that meet the visibility requirement'),
            'interval': {'low_pct': _pct(s.get('ci_low')), 'high_pct': _pct(s.get('ci_high')),
                         'level': s.get('ci_level'), 'method': s.get('ci_method'),
                         'wilson_low_pct': _pct(s.get('wilson_low')), 'wilson_high_pct': _pct(s.get('wilson_high'))},
            'figures': {'grasps': s.get('grasps'), 'feasible': s.get('feasible'), 'evaluated': s.get('evaluated'),
                        'meet': s.get('meets'), 'of_all_grasps_pct': _pct(s.get('cvs_of_all')),
                        'infeasible': compact(s.get('infeasible'))},
            'requirement': s.get('requirement'),
            'best_move': {'text': bm.get('text'), 'delta_pct': _pct(bm.get('delta_cvs'))} if bm else None,
            'rows': compact(doc.get('cameras'), items=8, max_depth=2),
            'grasp_set': compact(doc.get('grasp_set'), max_depth=1)}


def _teleop(doc):
    s = doc.get('summary') or {}
    return {**_head(doc, 'summary'), 'figure': _fig(s.get('teleoperable_pct'), '%',
                                                    'task volume teleoperable without a dead zone'),
            'figures': {k: _round(v) if isinstance(v, float) else v for k, v in s.items()
                        if k not in ('status', 'status_word', 'word', 'advice')},
            'devices': compact(doc.get('devices'), max_depth=2), 'bimanual': compact(doc.get('bimanual'), max_depth=2),
            'thresholds_note': doc.get('thresholds_note')}


def _cables(doc):
    s = doc.get('summary') or {}
    rows = [{'cable': c.get('name'), 'cable_uid': c.get('uid'), 'sensor': c.get('sensor'), 'status': c.get('status'),
             'required_m': _round(c.get('required_m')), 'suggested_m': _round(c.get('suggested_m')),
             'laid_m': _round(c.get('laid_m')), 'length_m': _round(c.get('length_m')),
             'min_bend_radius_m': _round(c.get('min_bend_radius')), 'flags': compact(c.get('flags'), max_depth=2)}
            for c in doc.get('cables') or []]
    return {**_head(doc, 'summary'), 'figure': _fig(s.get('clear'), 'cables', f'of {s.get("cables")} clear'),
            'flags': compact(s.get('flags')), 'suggested': bool(doc.get('suggested')), 'rows': rows}


def _bandwidth(doc):
    h = doc.get('headline') or {}
    disk = doc.get('disk') or {}
    rows = [{'sensor': s.get('name'), 'sensor_uid': s.get('uid'), 'interface': s.get('interface'),
             'payload_mbps': _round(s.get('payload_mbps')), 'wire_mbps': _round(s.get('wire_mbps')),
             'link_status': s.get('link_status'), 'link_load': _round(s.get('link_load')),
             'power_w': _round(s.get('power_w')), 'power_status': s.get('power_status'),
             'not_stated': compact(s.get('not_stated'))}
            for s in doc.get('sensors') or []]
    return {**_head(doc, 'headline'), 'figure': _fig(h.get('busiest_load_pct'), '%', 'load of the busiest link'),
            'figures': {'total_payload_mbps': _round(h.get('total_payload_mbps')), 'busiest': h.get('busiest'),
                        'gb_per_hour': _round(disk.get('gb_per_hour')),
                        'gb_per_episode': _round(disk.get('gb_per_episode')),
                        'write_needed_mb_s': _round(disk.get('write_needed_mb_s')),
                        'write_mb_s': _round(disk.get('write_mb_s'))},
            'findings': [{'severity': f.get('severity'), 'message': short(f.get('message', ''), 300)}
                         for f in doc.get('findings') or []][:8], 'rows': rows, 'units': doc.get('units')}


def _time_sync(doc):
    s = doc.get('summary') or {}
    rows = [{'stream': x.get('name'), 'sensor_uid': x.get('sensor_uid'), 'fps': x.get('fps'),
             'shutter': x.get('shutter'),
             'clock': x.get('clock'), 'trigger': x.get('trigger'), 'worst_skew_s': _round(x.get('worst_skew_s'), 6),
             'skew_low_s': _round(x.get('skew_low_s'), 6), 'complete': x.get('complete'), 'word': x.get('word')}
            for x in doc.get('streams') or []]
    return {**_head(doc, 'summary'),
            'figure': _fig(_round(s.get('worst_skew_s'), 6), 's', 'worst camera-to-joint skew'),
            'figures': {'worst_stream': s.get('worst_name'), 'skew_interval_s': compact(s.get('skew_interval_s')),
                        'label_error_m': _round(s.get('label_error_m'), 6), 'complete': s.get('complete'),
                        'skew_limit_s': doc.get('skew_limit_s'), 'skew_limit_default': doc.get('skew_limit_default')},
            'rows': rows, 'pairs': compact(doc.get('pairs'), max_depth=2),
            'joint_states': compact(doc.get('joint_states'), max_depth=1)}


def _vla_tokens(doc):
    s = doc.get('summary') or {}
    return {**_head(doc, 'summary'),
            'figure': _fig(s.get('fewest'), 'tokens', 'fewest ViT tokens on a critical object'),
            'fingertips_fewest': compact(s.get('fingertips_fewest'), max_depth=1),
            'rows': compact(doc.get('policies'), items=6, max_depth=3), 'warnings': compact(doc.get('warnings')),
            'pose_source': doc.get('pose_source')}


def _action_chunk(doc):
    h = doc.get('headline') or {}
    rows = [{'policy': x.get('name'), 'policy_uid': x.get('uid'), 'entry': x.get('entry'), 'status': x.get('status'),
             'worst_mm': _round(1000.0 * x['worst_m'], 3) if isinstance(x.get('worst_m'), (int, float)) else None,
             'worst_kind': x.get('worst_kind'), 'delay_offset_s': x.get('delay_offset_s'),
             'delay_offset_source': x.get('delay_offset_source'), 'missing': compact(x.get('missing')),
             'recommendations': compact(x.get('recommendations'), max_depth=2)}
            for x in doc.get('policies') or []]
    return {**_head(doc, 'headline'), 'figure': _fig(h.get('worst_mm'), 'mm', 'worst gripper-close error'),
            'figures': {k: h.get(k) for k in ('policy', 'kind', 'overshoot_mm', 'air_grasp_mm', 'reaction_mm')},
            'rows': rows, 'refused': compact(doc.get('refused')), 'approach': compact(doc.get('approach'), max_depth=1)}


def _handeye_plan(doc):
    w = doc.get('worst') or {}
    nx = doc.get('next') or {}
    return {**_head(doc), 'figure': _fig(w.get('ratio'), '', f'worst predicted sigma over the requirement '
                                                             f'({w.get("axis") or "-"})'),
            'figures': {'mode': doc.get('mode'), 'poses_planned': doc.get('n_planned'),
                        'poses_asked': doc.get('n_asked'),
                        'determined': doc.get('determined'), 'requirement': compact(doc.get('requirement')),
                        'candidates': compact(doc.get('candidates'), max_depth=1)},
            'sigma': compact(doc.get('sigma')), 'camera': compact(doc.get('camera'), max_depth=1),
            'board': {k: _get(doc, 'board', k) for k in ('name', 'what', 'default')},
            'next_pose': {'what': nx.get('what'), 'worst_gain_pct': nx.get('worst_gain_pct')} if nx else None,
            'assumptions': doc.get('assumptions')}


def _audit(doc):
    s = doc.get('summary') or {}
    return {**_head(doc, 'summary'), 'figure': _fig(s.get('pass_pct'), '%', 'episodes that pass'),
            'figures': {'episodes': s.get('episodes'), 'pass': s.get('pass'),
                        'drop_or_rerecord': compact(s.get('drop_or_rerecord')), 'fix': compact(s.get('fix')),
                        'recalibrate': compact(s.get('recalibrate'))},
            'requirements': compact(doc.get('requirements'), items=12, max_depth=2),
            'slips': compact(doc.get('slips'), max_depth=2)}


def _generic(doc):
    out = {**_head(doc, 'headline', 'summary')}
    for k in ('headline', 'summary'):
        if isinstance(doc.get(k), dict):
            out[k] = compact(doc[k], max_depth=2)
    return out


BRIEFS = {'coverage': _coverage, 'depth_validity': _depth_validity, 'grasp_visibility': _grasp_visibility,
          'teleop': _teleop, 'cables': _cables, 'bandwidth': _bandwidth, 'time_sync': _time_sync,
          'vla_tokens': _vla_tokens, 'action_chunk_sync': _action_chunk, 'handeye_plan': _handeye_plan,
          'audit': _audit}


def brief(sid: str, doc: dict) -> dict:
    """The figures of study sid's result doc (module docstring)."""
    from ..core import jobs
    info = jobs.STUDIES.get(sid)
    out = {'study': sid, 'title': info.title if info else sid}
    out.update(BRIEFS.get(sid, _generic)(doc))
    for k, v in _common(doc).items():
        out.setdefault(k, v)
    out['seconds'] = doc.get('seconds')
    out['cached'] = bool(doc.get('cached'))
    if doc.get('cancelled'):
        out['cancelled'] = True
    return out
