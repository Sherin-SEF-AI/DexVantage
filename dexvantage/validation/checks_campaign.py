"""Checks of the data-collection campaign planner (core/campaign.py) against references that do not share its code.

- campaign.williams_square: the rows core/campaign.williams builds for orders 1 to 12 against the definition of a
  Williams balanced Latin square (Williams 1949), counted here by brute force: every row a permutation, every column
  a permutation, every ordered pair of distinct conditions adjacent exactly once (even order; twice in the 2n rows of
  an odd order).
- campaign.schedule_balance: 40 seeded random campaigns, their balance counted from the exported CSV rows (parsed
  with Python's csv module, not the planner's own lists): every condition the same number of episodes per operator;
  every condition at every position of a round equally often when the design says position balance is exact; every
  ordered pair adjacent equally often within the rounds when it says carry-over balance is exact.
- campaign.session_limits: the same campaigns' sessions against the limits they were given, re-checked from the
  session times: no session over the fatigue limit, no day over its session count or its hours, no operator over
  their sessions per day, breaks kept, sessions on a day not overlapping, a session's length at least its episodes,
  resets and changeovers, and every planned episode scheduled exactly once.
- campaign.storage_vs_bandwidth: storage per episode and in total on the UR5e and dex-hand samples against the
  bandwidth study's own disk figures (core/studies/bandwidth.py: its gb_per_episode and gb_per_campaign at the
  campaign's episode length, episode count and recording ratio). The campaign reads those figures, so that part is a
  consistency check; the independent part adds the GB per episode up again here, Σ width × height × fps × bits per
  pixel of each selected stream (the stream's size, rate and format from its id, its bits per pixel from the library
  entry file read with json) × (1 + container overhead) × ratio × episode_s / 8 / 10⁹, a stream without stated bits
  per pixel adding nothing.
- campaign.ics_roundtrip: the calendar file parsed back by a strict RFC 5545 reader written here (CRLF line ends,
  lines of at most 75 octets, unfolding, BEGIN/END nesting, the required properties PRODID, VERSION, UID, DTSTAMP in
  UTC and DTSTART, TEXT unescaping) against the events the plan states (UID, start, end, summary, description).
- campaign.power_exact: the demonstrations per condition of the power analysis against the exact power of the test
  they are for, enumerated over both binomial distributions with SciPy: the pooled two-sided z-test without the
  continuity correction, the Yates-corrected one with it.
"""
from __future__ import annotations

import csv
import datetime as _dt
import io
import json
import math
import re

import numpy as np

from .suite import Outcome, check


# ====================================================================== the Williams square
@check('campaign.williams_square', 'campaign', 'Williams balanced Latin squares of order 1 to 12',
       'violations of the definition (rows and columns permutations, each ordered pair adjacent equally often)',
       'the definition of a Williams square (Williams 1949), counted by brute force', 0, 'count',
       refs=('williams1949',))
def _williams(ctx):
    from ..core.campaign import williams
    bad, detail = 0, {}
    for n in range(1, 13):
        rows = williams(n)
        want_rows = n if n % 2 == 0 or n == 1 else 2 * n
        v = int(len(rows) != want_rows)
        v += sum(sorted(r) != list(range(n)) for r in rows)
        for sq in (rows[:n], rows[n:2 * n]) if n % 2 and n > 1 else (rows,):
            v += sum(sorted(r[j] for r in sq) != list(range(n)) for j in range(n))
        pairs = {}
        for r in rows:
            for a, b in zip(r, r[1:]):
                pairs[(a, b)] = pairs.get((a, b), 0) + 1
        want = 1 if n % 2 == 0 else 2
        v += sum(pairs.get((a, b), 0) != want for a in range(n) for b in range(n) if a != b)
        detail[str(n)] = v
        bad += v
    return Outcome(float(bad), {'violations_by_order': detail})


# ====================================================================== random campaigns
def _random_specs(rng, n):
    out = []
    for _ in range(n):
        n_tasks, n_light = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        ep = [float(rng.uniform(5, 120)) for _ in range(n_tasks)]
        reset = float(rng.uniform(0, 60))
        chg = float(rng.uniform(0, 10))
        every = str(rng.choice(['day', 'session']))
        cal = float(rng.uniform(0, 15))
        need = math.ceil((chg * 60 + (cal * 60 if every == 'session' else 0) + max(ep) + reset) / 60) + 1
        sess = int(rng.integers(need, need + 120))
        day_h = float(rng.uniform((sess + cal) / 60 + 0.2, 10.5))
        out.append({
            'operators': [f'Op {k + 1}' for k in range(int(rng.integers(1, 5)))],
            'tasks': [{'name': f'Task {k + 1}', 'episode_s': ep[k]} for k in range(n_tasks)],
            'lighting': [{'name': f'Light {k + 1}'} for k in range(n_light)],
            'demos': {'mode': 'entered', 'per_condition': int(rng.integers(1, 31))},
            'keep_rate': float(rng.choice([1.0, 0.9, 0.75])),
            'reset_s': reset, 'changeover_min': chg, 'calibration': {'every': every, 'minutes': cal},
            'session_max_min': sess, 'break_min': int(rng.integers(0, 21)),
            'sessions_per_day': int(rng.integers(1, 7)), 'operator_sessions_per_day': int(rng.integers(1, 4)),
            'day_start': f'{int(rng.integers(6, 11)):02d}:{int(rng.choice([0, 15, 30, 45])):02d}', 'day_hours': day_h,
            'seed': int(rng.integers(0, 10 ** 6))})
    return out


def _plans(ctx, cid):
    key = ('campaign_plans', ctx.quick)
    if key not in ctx.cache:
        from ..core import campaign as CP
        from ..core.project import blank_project
        p = blank_project('Campaign check')
        rng = ctx.rng('campaign.random_specs')
        specs = _random_specs(rng, 16 if ctx.quick else 40)
        ctx.cache[key] = [(sp, CP.plan(p, sp)) for sp in specs]
    return ctx.cache[key]


@check('campaign.schedule_balance', 'campaign', 'Schedule balance over operators and positions',
       'conditions recorded unequally per operator, or (where the design says exact) per position or after another '
       'condition', 'counts from the exported CSV rows, parsed with the csv module', 0, 'count',
       refs=('williams1949',))
def _balance(ctx):
    from ..core import campaign as CP
    bad, n_pos, n_carry = 0, 0, 0
    for sp, doc in _plans(ctx, 'balance'):
        rows = list(csv.DictReader(io.StringIO(CP.to_csv(doc))))
        conds = [c['id'] for c in doc['conditions']]
        per = {}
        rounds = {}
        for r in rows:                                   # in session order: a round's blocks in time order
            per[(r['operator'], r['condition'])] = per.get((r['operator'], r['condition']), 0) + int(r['episodes'])
            seq = rounds.setdefault((r['operator'], int(r['round'])), [])
            if not seq or seq[-1] != r['condition']:
                seq.append(r['condition'])
        counts = {per.get((o, c), 0) for o in sp['operators'] for c in conds}
        bad += int(len(counts) != 1)
        bad += sum(sorted(s) != sorted(conds) for s in rounds.values())     # a round is every condition once
        if doc['design']['position_balance'] == 'exact':
            n_pos += 1
            for p in range(len(conds)):
                bad += int(len({sum(1 for s in rounds.values() if s[p] == c) for c in conds}) != 1)
        if doc['design']['carryover_balance'] == 'exact' and len(conds) > 1:
            n_carry += 1
            pairs = {}
            for s in rounds.values():
                for a, b in zip(s, s[1:]):
                    pairs[(a, b)] = pairs.get((a, b), 0) + 1
            bad += int(len({pairs.get((a, b), 0) for a in conds for b in conds if a != b}) != 1)
    return Outcome(float(bad), {'campaigns': len(_plans(ctx, 'balance')), 'position_exact': n_pos,
                                'carryover_exact': n_carry})


@check('campaign.session_limits', 'campaign', 'Sessions within the stated limits',
       'sessions or days over a stated limit, overlapping or too short, and episodes not scheduled exactly once',
       'the limits each random campaign was given, re-checked from the session times', 0, 'count')
def _limits(ctx):
    bad, sessions = 0, 0
    for sp, doc in _plans(ctx, 'limits'):
        st = doc['settings']
        h, m = (int(x) for x in st['day_start'].split(':'))
        d0 = h * 60 + m
        d1 = d0 + round(st['day_hours'] * 60)
        by_c = {c['id']: c for c in doc['conditions']}
        days = {}
        for s in doc['sessions']:
            sessions += 1
            bad += int(s['end_min'] - s['start_min'] > st['session_max_min'])
            bad += int(s['start_min'] < d0 or s['end_min'] > d1)
            need = sum(b['episodes'] * (by_c[b['condition']]['episode_s'] + st['reset_s']) for b in s['blocks'])
            changes = 1 + sum(a['condition'] != b['condition'] for a, b in zip(s['blocks'], s['blocks'][1:]))
            need += changes * st['changeover_min'] * 60
            if st['calibration']['every'] == 'session':
                need += st['calibration']['minutes'] * 60
            bad += int(s['end_min'] - s['start_min'] + 1e-9 < need / 60.0)
            days.setdefault(s['day'], []).append(s)
        for d, ss in days.items():
            bad += int(len(ss) > st['sessions_per_day'])
            for o in set(s['operator'] for s in ss):
                bad += int(sum(s['operator'] == o for s in ss) > st['operator_sessions_per_day'])
            ss = sorted(ss, key=lambda s: s['start_min'])
            bad += sum(b['start_min'] < a['end_min'] + st['break_min'] for a, b in zip(ss, ss[1:]))
            if st['calibration']['every'] == 'day':
                bad += int(ss[0]['start_min'] < d0 + math.ceil(st['calibration']['minutes'] - 1e-9))
        want = sum(c['recorded'] for c in doc['conditions'])
        got = sum(b['episodes'] for s in doc['sessions'] for b in s['blocks'])
        bad += int(want != got)
        target = doc['demos']['target']
        bad += sum(c['recorded'] * st['keep_rate'] + 1e-9 < target for c in doc['conditions'])
    return Outcome(float(bad), {'campaigns': len(_plans(ctx, 'limits')), 'sessions': sessions})


# ====================================================================== storage
@check('campaign.storage_vs_bandwidth', 'campaign', 'Storage against the bandwidth study\'s disk figures',
       'largest relative difference of GB per episode and GB in total', 'the bandwidth study\'s gb_per_episode and '
       'gb_per_campaign (core/studies/bandwidth.py) at the same episode length, count and ratio, and GB per episode '
       'added up again here from the selected streams\' sizes, rates and the library files\' bits per pixel',
       1e-9, 'relative')
def _storage(ctx):
    from ..core import campaign as CP
    from ..core import library as LB
    from ..core.robot.samples import sample_project
    from ..core.studies import bandwidth as BW
    worst, rows = 0.0, []
    for name, fmt, ratio in (('ur5e', 'raw', None), ('dexhand', 'lerobot', 0.08)):
        p = sample_project(name)
        sp = {'demos': {'per_condition': 7}, 'operators': ['A', 'B'], 'episode_s': 42.0, 'format': fmt}
        if ratio:
            sp['compression_ratio'] = {'value': ratio, 'source': 'check'}
        doc = CP.plan(p, sp)
        c = doc['conditions'][0]
        q = p.snapshot()
        q.analysis.studies['bandwidth'] = {'episode_s': 42.0, 'episodes': c['recorded'], 'record_ratio': ratio or 1.0}
        disk = BW.budget(q)['disk']
        e1 = abs(c['gb_per_episode'] - disk['gb_per_episode']) / max(disk['gb_per_episode'], 1e-300)
        e2 = abs(doc['totals']['storage_gb'] - disk['gb_per_campaign']) / max(disk['gb_per_campaign'], 1e-300)
        # the total is rounded to 3 decimals in the result: compare at that resolution
        e2 = 0.0 if abs(doc['totals']['storage_gb'] - disk['gb_per_campaign']) <= 5e-4 + 1e-12 else e2
        bst = BW.settings_of(q)
        hand_mbps = 0.0
        links = (q.rig if isinstance(q.rig, dict) else {}).get('links') or []
        for s in (s for s in q.variant.sensors if s.enabled and s.model_id):
            e = LB.get(s.model_id)
            if e is None or not e.path:
                continue
            with open(e.path, encoding='utf-8') as fh:
                modes = json.load(fh).get('streams') or []
            ids = next((k.get('stream_ids') or [] for k in links if k.get('sensor_uid') == s.uid), [])
            for r in BW.sensor_streams(s, e, ids, bst)['streams']:
                if r.get('extra'):                     # a stream the user states a rate for (an IMU, audio)
                    hand_mbps += float(r.get('payload_mbps') or 0.0)
                    continue
                imager, size, fps, fmt = r['id'].split('|')
                w, h = (int(x) for x in size.split('x'))
                bpp = next((m.get('bits_per_pixel') for m in modes if m.get('imager') == imager
                            and int(m.get('width', 0)) == w and int(m.get('height', 0)) == h
                            and (m.get('format') or '') == fmt), None)
                if isinstance(bpp, (int, float)):
                    hand_mbps += w * h * float(fps) * bpp / 1e6
        hand = hand_mbps * (1 + float(bst['container_overhead'])) * (ratio or 1.0) * 42.0 / 8 / 1000
        e3 = abs(c['gb_per_episode'] - hand) / max(hand, 1e-300)
        worst = max(worst, e1, e2, e3)
        rows.append({'sample': name, 'gb_per_episode': c['gb_per_episode'], 'bandwidth': disk['gb_per_episode'],
                     'by_hand': hand,
                     'total_gb': doc['totals']['storage_gb'], 'bandwidth_total': disk['gb_per_campaign'],
                     'incomplete': doc['totals']['storage_incomplete']})
    return Outcome(worst, {'rows': rows})


# ====================================================================== the calendar
class IcsError(ValueError):
    pass


def parse_ics(data: bytes) -> dict:
    """A strict reader of the subset of RFC 5545 a plan uses: {'calendar': {prop: value}, 'events': [{prop: value}]}
    with TEXT unescaped and DATE-TIME values as naive datetimes (DTSTAMP: aware UTC). Raises IcsError on CR or LF
    alone, a line over 75 octets, a property outside a component, unbalanced BEGIN/END, a repeated or missing
    required property, or a malformed value."""
    if not data.endswith(b'\r\n'):
        raise IcsError('the file does not end with CRLF')
    raw = data[:-2].split(b'\r\n')
    for ln in raw:
        if b'\r' in ln or b'\n' in ln:
            raise IcsError('a bare CR or LF')
        if len(ln) > 75:
            raise IcsError(f'a line of {len(ln)} octets')
    lines = []
    for ln in raw:
        if ln[:1] in (b' ', b'\t'):
            if not lines:
                raise IcsError('a continuation line first')
            lines[-1] += ln[1:]
        else:
            lines.append(ln)
    text = [ln.decode('utf-8') for ln in lines]
    stack, cal, events, cur = [], {}, [], None
    for ln in text:
        m = re.match(r'^([A-Za-z0-9-]+)((?:;[^:]*)?):(.*)$', ln)
        if not m:
            raise IcsError(f'not a content line: {ln[:40]}')
        name, params, value = m.group(1).upper(), m.group(2), m.group(3)
        if name == 'BEGIN':
            stack.append(value)
            if value == 'VEVENT':
                if stack != ['VCALENDAR', 'VEVENT']:
                    raise IcsError('a VEVENT outside VCALENDAR')
                cur = {}
            elif value != 'VCALENDAR' or len(stack) != 1:
                raise IcsError(f'unexpected BEGIN:{value}')
            continue
        if name == 'END':
            if not stack or stack[-1] != value:
                raise IcsError(f'END:{value} does not close {stack[-1] if stack else "anything"}')
            stack.pop()
            if value == 'VEVENT':
                for req in ('UID', 'DTSTAMP', 'DTSTART'):
                    if req not in cur:
                        raise IcsError(f'a VEVENT without {req}')
                events.append(cur)
                cur = None
            continue
        if not stack:
            raise IcsError(f'{name} outside a component')
        target = cur if stack[-1] == 'VEVENT' else cal
        if name in target:
            raise IcsError(f'{name} repeated')
        if params:
            raise IcsError(f'{name} has parameters this reader does not expect')
        if name in ('DTSTART', 'DTEND'):
            if not re.fullmatch(r'\d{8}T\d{6}', value):
                raise IcsError(f'{name} is not a floating DATE-TIME: {value}')
            target[name] = _dt.datetime.strptime(value, '%Y%m%dT%H%M%S')
        elif name == 'DTSTAMP':
            if not re.fullmatch(r'\d{8}T\d{6}Z', value):
                raise IcsError('DTSTAMP is not in UTC')
            target[name] = _dt.datetime.strptime(value, '%Y%m%dT%H%M%SZ').replace(tzinfo=_dt.timezone.utc)
        elif name in ('SUMMARY', 'DESCRIPTION', 'X-WR-CALNAME'):
            out, i = [], 0
            while i < len(value):
                ch = value[i]
                if ch == '\\':
                    nx = value[i + 1:i + 2]
                    if nx in ('\\', ';', ','):
                        out.append(nx)
                    elif nx in ('n', 'N'):
                        out.append('\n')
                    else:
                        raise IcsError(f'a bad escape \\{nx}')
                    i += 2
                    continue
                if ch in (';', ','):
                    raise IcsError(f'an unescaped {ch} in {name}')
                out.append(ch)
                i += 1
            target[name] = ''.join(out)
        else:
            target[name] = value
    if stack:
        raise IcsError('BEGIN without END')
    for req in ('PRODID', 'VERSION'):
        if req not in cal:
            raise IcsError(f'the calendar has no {req}')
    if cal['VERSION'] != '2.0':
        raise IcsError('VERSION is not 2.0')
    return {'calendar': cal, 'events': events}


@check('campaign.ics_roundtrip', 'campaign', 'iCalendar file read back',
       'events whose UID, start, end, summary or description differ, plus format violations',
       'a strict RFC 5545 reader written here (folding, CRLF, nesting, required properties, TEXT escapes)', 0,
       'count', refs=('rfc5545',))
def _ics(ctx):
    from ..core import campaign as CP
    bad, n = 0, 0
    stamp = _dt.datetime(2026, 11, 7, 12, 0, tzinfo=_dt.timezone.utc)
    plans = _plans(ctx, 'ics')[:8]
    for k, (sp, doc) in enumerate(plans):
        doc = dict(doc, name=f'Check; campaign, {k} \\ «ü»')        # characters that must be escaped or folded
        start = '2026-11-12' if k % 2 else None
        data = CP.to_ics(doc, start_date=start or '2026-11-09', dtstamp=stamp).encode('utf-8')
        try:
            got = parse_ics(data)
        except IcsError:
            bad += 1
            continue
        want = CP.ics_events(doc, start or '2026-11-09')
        n += len(want)
        bad += abs(len(got['events']) - len(want))
        for g, w in zip(got['events'], want):
            bad += int(g.get('UID') != w['uid'] or g.get('DTSTART') != w['start'] or g.get('DTEND') != w['end']
                       or g.get('SUMMARY') != w['summary'] or g.get('DESCRIPTION') != w['description']
                       or g.get('DTSTAMP') != stamp)
        bad += int(len({e['UID'] for e in got['events']}) != len(got['events']))
    return Outcome(float(bad), {'calendars': len(plans), 'events': n})


# ====================================================================== the power analysis
POWER_CASES = ((0.5, 0.7), (0.3, 0.5), (0.6, 0.8), (0.7, 0.9), (0.2, 0.4), (0.4, 0.6), (0.5, 0.65), (0.8, 0.95),
               (0.1, 0.3), (0.5, 0.8), (0.6, 0.75))


def exact_power(n: int, p1: float, p2: float, alpha: float, continuity: bool) -> float:
    """The exact power of the two-sided pooled z-test of two proportions with n per group (with continuity: the
    Yates-corrected statistic, |p̂1 − p̂2| − 1/n), enumerating both binomial distributions (SciPy)."""
    from scipy.stats import binom, norm
    x = np.arange(n + 1)
    a, b = binom.pmf(x, n, p1), binom.pmf(x, n, p2)
    X1, X2 = np.meshgrid(x, x, indexing='ij')
    pb = (X1 + X2) / (2.0 * n)
    se = np.sqrt(pb * (1 - pb) * 2.0 / n)
    d = np.abs(X1 - X2) / n
    if continuity:
        d = np.maximum(d - 1.0 / n, 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.where(se > 0, d / se, 0.0)
    return float((np.outer(a, b) * (z > norm.ppf(1 - alpha / 2))).sum())


@check('campaign.power_exact', 'campaign', 'Demonstrations per condition against the exact power',
       'largest |exact power at the planned n − the target power| over 11 pairs of success rates, 80 % and 90 % '
       'power, with and without the continuity correction',
       'the exact power of the test the formula is for, by enumeration over both binomial distributions (SciPy)',
       0.04, 'power (0-1)', refs=('fleiss2003', 'fleiss_tytun_ury1980', 'scipy_stats'))
def _power(ctx):
    from ..core.campaign import power_n
    worst, rows = 0.0, []
    cases = POWER_CASES[:4] if ctx.quick else POWER_CASES
    for cc in (False, True):
        for p1, p2 in cases:
            for pw in (0.8, 0.9):
                n = power_n(p1, p2, 0.05, pw, cc)['n']
                e = exact_power(n, p1, p2, 0.05, cc)
                worst = max(worst, abs(e - pw))
                rows.append({'p1': p1, 'p2': p2, 'power': pw, 'continuity': cc, 'n': n, 'exact': round(e, 4)})
    return Outcome(worst, {'cases': rows},
                   note='The normal approximation is furthest off near a rate of 0 or 1 (0.8 against 0.95: exact '
                        'power 0.833 for 80 %); the exact power of a discrete test also steps with n.')
