"""The data-collection campaign planner: which demonstrations to record, by whom, in what order, on which days, how
long it takes and how much it stores, with a checklist for every session and the plan as CSV, iCalendar and a
printable page.

    settings_of(project, spec=None) -> dict      the spec with every default filled in ('defaults' lists them)
    read_spec(path_or_text) -> dict              a spec file (schema SPEC_SCHEMA) or JSON text
    plan(project, spec=None, ctx=None) -> dict   the plan (the study 'campaign' returns it)
    williams(n) -> [[int]]                       the rows of a Williams balanced Latin square of order n
    power_n(p1, p2, alpha, power, continuity)    demonstrations per condition for a difference in success rate
    storage(project, variant_uid, st) -> dict    GB per second of recording from the bandwidth study's data rates
    to_csv(doc), to_ics(doc, start_date, dtstamp), to_html(doc)   the exports (text)
    write_files(doc, base) -> [paths]            <base>.csv, <base>.ics, <base>.html
    apply_plan(project, doc)                     writes the plan into project.analysis.studies['campaign']['plan']
    balance(doc) -> dict                         counts per operator, per position and per ordered pair (the checks)

Ported in part from Vantage core/simcampaign.py (the pattern of a seeded design whose every run is keyed and kept, the
per-seed shuffle of each condition list) and core/fleet.py (settings clamped to LIMITS, dates parsed and normalised);
nothing automotive is kept: there are no drives, vehicles or weathers here.

Inputs
------
The spec lives in project.analysis.studies['campaign'] (or a spec file for batch mode: `--campaign SPEC.json`). Every
key is optional; DEFAULTS holds the defaults, each a DexVantage default, labelled as such in the result:

* Conditions: the cartesian product of the levels of five axes (or an explicit list, 'conditions'):
  - tasks: [{'name', 'trajectory' (uid or name), 'episode_s'}]; default one task per coordinated motion of
    project.tasks.trajectories (trajectories of different robots over the same time span are one bimanual motion),
    else one per grasp set, else one task called "Task";
  - variants: rig variants (uid or name); default the active variant;
  - lighting: presets [{'name', 'lights': {light uid or name: dim 0..1, 0 = off}}]; default one preset, the
    workspace's lights as they are;
  - object_sets: [{'name', 'objects': [object uid or name]}]; default one set, every task object;
  - devices: [{'name', 'devices': [teleop device uid or name]}]; default one level, every teleop device.
* operators: names (default one, "Operator 1"); each operator records every condition equally often.
* Demonstrations per condition: 'demos' {'mode': 'entered', 'per_condition': 50 (default)} or {'mode': 'power',
  'p_base', 'p_target', 'alpha' (0.05), 'power' (0.8), 'continuity' (true)}: the number of demonstrations per
  condition that tells a success rate p_base from p_target with the two-sided test at alpha with that power (below).
  keep_rate (default 1: every episode kept) scales what is recorded so the expected kept episodes reach the target.
* Times: episode_s (the task's trajectory duration when it has one, else 60 s, a default), reset_s (30 s),
  changeover_min (5 min: setting a condition up at the start of a session and at each change of condition),
  calibration {'every': 'day' | 'session' (default day), 'minutes' (10), 'check_poses' (5)}, break_min (15).
* Operator fatigue limits: session_max_min (90 min of recording in one session) and operator_sessions_per_day (2);
  the rig: sessions_per_day (4), day_start ('09:00'), day_hours (8); start_date (ISO date, none by default: days are
  numbered) and weekdays_only (true).
* Storage: format 'raw' | 'lerobot' | 'rosbag' (RECORDING_FORMATS) and compression_ratio {'value', 'source'}.
* Budget: {'disk_gb', 'days', 'upload_mbps', 'upload_hours_per_day', 'operator_hours'}; none by default.

The design
----------
C conditions, O operators. Each operator records each condition m times, in B rounds; in each round the operator
records every condition once, as a block of k = m / B consecutive episodes, in the order of one row of a **Williams
balanced Latin square** (Williams 1949): for even C, the C rows 0, 1, C−1, 2, C−2, … shifted cyclically, in which
every condition is at every position once and follows every other condition exactly once; for odd C, those C rows and
their C mirror images (2C rows), in which every condition follows every other exactly twice. The S = O × B sequences
take the square's rows in turn (row s mod L, L its row count); the seed shuffles which condition each symbol of the
square stands for and which sequence (operator, round) gets which row. Then:

* every condition appears exactly B times per operator (always: each round is a permutation of the conditions);
* every condition appears equally often at every position when S is a multiple of C (whole squares are used);
* every ordered pair of conditions is adjacent equally often (first-order carry-over balance) when S is a multiple
  of L.

rounds 'auto' (the default) takes the smallest B ≤ m with S a multiple of L, else of C, else B = 1 (the result
says which balance holds and what it would need); m is rounded up to a multiple of B so all blocks are equal. What is
recorded per condition is O × m, at least the target divided by keep_rate.

Sessions and days. An operator's episodes are packed in order into sessions: a session holds the changeovers and the
episodes (episode_s + reset_s each) that fit within session_max_min (whole minutes, rounded up); an episode never
straddles two sessions. The sessions are then evened out: the packing is repeated with the shortest limit that needs
no more sessions (found by bisection: in-order packing never needs more sessions for a longer limit), so the last
session of an operator is not a short remainder. Days are filled in passes: each pass offers one session to every
operator who has one left, in an order rotated by one operator each day (who records first changes from day to day),
while the day has a session slot left (sessions_per_day), the operator is under operator_sessions_per_day and the
session ends within the day (day_start + day_hours, after the calibration check when it is daily and a break_min
break after each session).
Storage per episode is Σ over the enabled sensors of the condition's variant of the bandwidth study's data rate
(core/studies/bandwidth.py: width × height × fps × bits per pixel of the stated stream modes, or your measured rate) ×
the format's ratio × (1 + the container overhead of the bandwidth study's settings) × episode_s / 8 / 1000 GB: the
bandwidth study's own disk arithmetic.

Demonstrations by power analysis. The per-group size to tell p1 from p2 with a two-sided test at level α with power
1 − β (Fleiss, Levin & Paik 2003, the normal approximation):

    n = [z_{1−α/2} √(2 p̄ q̄) + z_{1−β} √(p1 q1 + p2 q2)]² / (p2 − p1)²,      p̄ = (p1 + p2)/2,  q = 1 − p

and with the continuity correction (Fleiss, Tytun & Ury 1980): n_c = n/4 · (1 + √(1 + 4 / (n |p2 − p1|)))². When
core/stats.py is there, its n_two_proportions (normal method: the same formula) gives the figure and the power it
reaches at n; the formula here is the fallback.

Recording formats. 'raw' (the frames as streamed: ratio 1), 'rosbag' (ROS 2 bag of raw image messages: ratio 1), and
'lerobot' (a LeRobot dataset: the videos encoded as core/datasets/lerobot.py writes them, AV1 with libsvtav1, yuv420p,
g 2, crf 30, preset 12 at v3.0, LeRobot's encoder defaults). No source states the size an encoder gives (it depends on
the content), so the lerobot format uses your compression_ratio, measured on a recorded episode; without it the size
is the raw size and the result says so.

Status against the budget: each figure the budget names (storage against disk_gb, days, the busiest day's upload time
against upload_hours_per_day at upload_mbps, the busiest operator's recording hours) is within (good), tight (above
TIGHT = 90 % of the limit, a default: warning) or over (critical); no budget figure: warning, "no budget set".

Known limits
------------
The schedule assumes every episode takes episode_s + reset_s: real demonstrations vary, fail and are re-done (the
keep rate covers discarded episodes on average, not their time); a task's trajectory duration is the robot's
scripted motion, which a person teleoperating may be slower than; operators are taken as interchangeable in speed;
the fatigue limits are defaults, not ergonomics findings; position and carry-over balance are exact only for the
numbers of sequences the design states (else partial, as the result says), and the balance is of blocks, not of
single episodes; conditions are not balanced across days (the design balances them per operator and per position;
the operators' order rotates from day to day, and a day holds what fits); days are filled greedily, which is not the
fewest days possible when sessions differ in length; the storage of a compressed format is the raw size unless you
enter a measured ratio; a sensor whose data rate is not stated adds nothing (the storage is then "at least"); upload
is a steady rate with no protocol overhead; times are local wall-clock times with no time zone (the calendar's
events are floating times, RFC 5545 §3.3.5); a calendar day is not checked against holidays. The power analysis is
for two independent proportions (two conditions compared by success rate); it does not cover several conditions at
once, paired designs, or learning-curve effects of the policy trained on the data.
"""
from __future__ import annotations

import copy
import csv
import datetime as _dt
import hashlib
import html
import io
import itertools
import json
import math
import os
from statistics import NormalDist

import numpy as np

from .jobs import StudyError

STUDY_ID = 'campaign'
SPEC_SCHEMA = 'dexvantage.campaign_spec/1'
PLAN_SCHEMA = 'dexvantage.campaign_plan/1'
TIGHT = 0.9                         # a default: a budget figure above this share of its limit is "tight"
MAX_CONDITIONS = 500
MAX_EPISODES = 200_000
MAX_ROUNDS = 64
AXES = ('task', 'variant', 'lighting', 'objects', 'device')
AXIS_TITLE = {'task': 'Task', 'variant': 'Rig variant', 'lighting': 'Lighting', 'objects': 'Objects',
              'device': 'Teleop device'}
WORDS = {'good': 'within budget', 'warning': 'tight', 'critical': 'over budget'}
FORMATS = ('raw', 'lerobot', 'rosbag')
FORMAT_TITLE = {'raw': 'Raw frames as streamed', 'lerobot': 'LeRobot dataset (encoded video)',
                'rosbag': 'ROS 2 bag (raw image messages)'}
CALIBRATION_EVERY = ('day', 'session')
# the power analysis goes through core/stats.py's n_two_proportions (its normal method, the same formula, with the
# power it reaches at n); the formula in power_n stays as the fallback when that module is missing or refuses
STATS_FUNCTION = 'n_two_proportions'

DEFAULTS = {
    'name': 'Campaign',
    'operators': ['Operator 1'],
    'demos': {'mode': 'entered', 'per_condition': 50, 'p_base': 0.5, 'p_target': 0.7, 'alpha': 0.05, 'power': 0.8,
              'continuity': True},
    'keep_rate': 1.0,
    'episode_s': None,               # None: each task's trajectory duration, else DEFAULT_EPISODE_S
    'reset_s': 30.0,
    'changeover_min': 5.0,
    'calibration': {'every': 'day', 'minutes': 10.0, 'check_poses': 5},
    'session_max_min': 90,
    'break_min': 15,
    'sessions_per_day': 4,
    'operator_sessions_per_day': 2,
    'day_start': '09:00',
    'day_hours': 8.0,
    'start_date': None,
    'weekdays_only': True,
    'rounds': 'auto',
    'seed': 0,
    'format': 'raw',
    'compression_ratio': None,
    'budget': {},
}
SPEC_KEYS = frozenset(DEFAULTS) | {'tasks', 'variants', 'lighting', 'object_sets', 'devices', 'conditions',
                                    'schema', 'provenance', 'plan'}     # what read_spec takes
DEFAULT_EPISODE_S = 60.0             # a default, as the bandwidth study's episode length
LIMITS = {'keep_rate': (0.01, 1.0), 'episode_s': (0.1, 86400.0), 'reset_s': (0.0, 86400.0),
          'changeover_min': (0.0, 600.0), 'session_max_min': (1, 1440), 'break_min': (0, 1440),
          'sessions_per_day': (1, 48), 'operator_sessions_per_day': (1, 48), 'day_hours': (0.1, 24.0),
          'seed': (0, 2 ** 31 - 1)}
DEMO_LIMITS = {'per_condition': (1, 100_000), 'p_base': (0.0, 1.0), 'p_target': (0.0, 1.0), 'alpha': (1e-6, 0.5),
               'power': (0.5, 0.9999)}
BUDGET_KEYS = {'disk_gb': ('Storage', 'GB'), 'days': ('Days', 'days'), 'upload_mbps': ('Upload link', 'Mbit/s'),
               'upload_hours_per_day': ('Upload time per day', 'h'), 'operator_hours': ('Recording per operator', 'h')}
REFERENCES = (
    'E. J. Williams, "Experimental designs balanced for the estimation of residual effects of treatments", Australian '
    'Journal of Scientific Research, Series A 2(2), 149-168, 1949, https://doi.org/10.1071/CH9490149 (read '
    '2026-11-07).',
    'J. L. Fleiss, B. Levin and M. C. Paik, Statistical Methods for Rates and Proportions, 3rd ed., Wiley, 2003 '
    '(sample size for comparing two independent proportions, the normal approximation).',
    'J. L. Fleiss, A. Tytun and H. K. Ury, "A simple approximation for calculating sample sizes for comparing '
    'independent proportions", Biometrics 36(2), 343-346, 1980, https://doi.org/10.2307/2529990 (read 2026-11-07).',
    'B. Desruisseaux (ed.), Internet Calendaring and Scheduling Core Object Specification (iCalendar), RFC 5545, '
    'September 2009, https://www.rfc-editor.org/rfc/rfc5545 (read 2026-11-07).',
)
ASSUMPTIONS = (
    'Every episode takes the episode time plus the reset time; failed and repeated demonstrations are not timed '
    '(the keep rate covers discarded episodes on average).',
    'Operators are taken as equally fast; the fatigue limits are DexVantage defaults until you set your own.',
    'Storage is the bandwidth study\'s data rate of the enabled sensors (stated stream modes or your measured rate) '
    'times the format\'s ratio and the container overhead; a compressed format\'s ratio is not stated by any source.',
    'Times are local wall-clock times without a time zone.',
)


class CampaignError(StudyError):
    """A campaign spec DexVantage cannot plan; the message says why in plain words."""


# ====================================================================== small helpers
def _num(v, default=None):
    if isinstance(v, bool):
        return default
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _clamp(v, lo, hi):
    return min(max(v, lo), hi)


def _r(v, d=4):
    return None if v is None or not math.isfinite(float(v)) else round(float(v), d)


def _hhmm(minutes: float) -> str:
    m = int(round(minutes))
    return f'{m // 60:02d}:{m % 60:02d}'


def parse_hhmm(text) -> int | None:
    """'09:30' -> 570 minutes after midnight; None when it is not a time of day."""
    try:
        h, m = str(text).strip().split(':')
        h, m = int(h), int(m)
    except (ValueError, AttributeError):
        return None
    return h * 60 + m if 0 <= h < 24 and 0 <= m < 60 else None


def parse_date(text) -> _dt.date | None:
    """An ISO date (2026-11-09); None when it is not one."""
    if isinstance(text, _dt.date):
        return text
    try:
        return _dt.date.fromisoformat(str(text).strip()[:10])
    except (ValueError, TypeError):
        return None


def _plural(n, word, plural=None):
    return f'{n:,} {word if n == 1 else (plural or word + "s")}'


# ====================================================================== the spec
def read_spec(src) -> dict:
    """A campaign spec: a dict, JSON text or a .json file (schema SPEC_SCHEMA when it names one)."""
    if isinstance(src, dict):
        doc = src
    else:
        text = str(src)
        if not text.lstrip().startswith(('{', '[')):
            try:
                with open(text, encoding='utf-8') as fh:
                    text = fh.read()
            except OSError as e:
                raise CampaignError(f'Could not read the campaign spec {os.path.basename(str(src))}: '
                                    f'{e.strerror or e}.') from None
        try:
            doc = json.loads(text)
        except ValueError as e:
            raise CampaignError(f'The campaign spec is not valid JSON ({e}).') from None
    if not isinstance(doc, dict):
        raise CampaignError('The campaign spec must be a JSON object, e.g. {"operators": ["A", "B"]}.')
    sch = doc.get('schema')
    if sch not in (None, SPEC_SCHEMA):
        raise CampaignError(f'The file is a "{sch}" document, not a campaign spec ({SPEC_SCHEMA}).')
    unknown = [k for k in doc if k not in SPEC_KEYS]
    if unknown:                                         # a misspelt key would otherwise leave its default in silence
        k = unknown[0]
        hint = (' The power analysis is "demos": {"mode": "power", "p_base": 0.5, "p_target": 0.7}.'
                if k in ('power', 'p1', 'p2', 'p_base', 'p_target', 'alpha', 'per_condition') else '')
        raise CampaignError(f'The campaign spec has no setting "{k}"; the settings are '
                            f'{", ".join(sorted(SPEC_KEYS - {"schema", "provenance", "plan"}))}.{hint}')
    return {k: v for k, v in doc.items() if k not in ('schema', 'provenance', 'plan')}


def settings_of(project, spec: dict | None = None) -> dict:
    """The stored spec (or spec) over DEFAULTS, clamped; 'defaults' lists the keys left at their default."""
    given = read_spec(spec) if spec is not None else {
        k: v for k, v in (project.analysis.studies.get(STUDY_ID) or {}).items() if k != 'plan'}
    st = copy.deepcopy(DEFAULTS)
    used = []
    st['name'] = str(given.get('name') or DEFAULTS['name'])
    ops = given.get('operators')
    if isinstance(ops, int) and not isinstance(ops, bool) and ops > 0:
        ops = [f'Operator {i + 1}' for i in range(min(ops, 100))]
    if isinstance(ops, list) and [str(o).strip() for o in ops if str(o).strip()]:
        names = []
        for o in ops:
            o = str(o).strip()
            if o and o not in names:
                names.append(o)
        st['operators'] = names[:100]
    else:
        used.append('operators')
    for k, (lo, hi) in LIMITS.items():
        v = _num(given.get(k))
        if v is None:
            if k != 'episode_s':
                used.append(k)
            continue
        st[k] = _clamp(v, lo, hi)
    for k in ('session_max_min', 'break_min', 'sessions_per_day', 'operator_sessions_per_day', 'seed'):
        st[k] = int(round(st[k]))
    d = given.get('demos') if isinstance(given.get('demos'), dict) else {}
    demos = dict(DEFAULTS['demos'])
    if d.get('mode') in ('entered', 'power'):
        demos['mode'] = d['mode']
    else:
        used.append('demos.mode')
    for k, (lo, hi) in DEMO_LIMITS.items():
        v = _num(d.get(k))
        if v is None:
            used.append(f'demos.{k}')
        else:
            demos[k] = _clamp(v, lo, hi)
    demos['per_condition'] = int(round(demos['per_condition']))
    if isinstance(d.get('continuity'), bool):
        demos['continuity'] = d['continuity']
    else:
        used.append('demos.continuity')
    st['demos'] = demos
    c = given.get('calibration') if isinstance(given.get('calibration'), dict) else {}
    cal = dict(DEFAULTS['calibration'])
    if c.get('every') in CALIBRATION_EVERY:
        cal['every'] = c['every']
    else:
        used.append('calibration.every')
    v = _num(c.get('minutes'))
    cal['minutes'] = _clamp(v, 0.0, 600.0) if v is not None else cal['minutes']
    if v is None:
        used.append('calibration.minutes')
    v = _num(c.get('check_poses'))
    cal['check_poses'] = int(_clamp(round(v), 1, 100)) if v is not None else cal['check_poses']
    if v is None:
        used.append('calibration.check_poses')
    st['calibration'] = cal
    t = parse_hhmm(given.get('day_start')) if given.get('day_start') is not None else None
    if t is None:
        used.append('day_start')
    else:
        st['day_start'] = _hhmm(t)
    sd = parse_date(given.get('start_date')) if given.get('start_date') else None
    st['start_date'] = sd.isoformat() if sd else None
    if isinstance(given.get('weekdays_only'), bool):
        st['weekdays_only'] = given['weekdays_only']
    else:
        used.append('weekdays_only')
    r = given.get('rounds')
    if isinstance(r, (int, float)) and not isinstance(r, bool) and r >= 1:
        st['rounds'] = int(min(round(r), MAX_ROUNDS))
    else:
        used.append('rounds')
    if given.get('format') in FORMATS:
        st['format'] = given['format']
    else:
        used.append('format')
    cr = given.get('compression_ratio')
    x = _num(cr.get('value') if isinstance(cr, dict) else cr)
    st['compression_ratio'] = ({'value': _clamp(x, 1e-4, 10.0), 'source': str(cr.get('source') or '')
                               if isinstance(cr, dict) else ''} if x is not None and x > 0 else None)
    b = given.get('budget') if isinstance(given.get('budget'), dict) else {}
    st['budget'] = {k: _num(b.get(k)) for k in BUDGET_KEYS if _num(b.get(k)) is not None and _num(b.get(k)) > 0}
    for k in ('tasks', 'variants', 'lighting', 'object_sets', 'devices', 'conditions'):
        st[k] = copy.deepcopy(given.get(k)) if isinstance(given.get(k), list) else []
    st['defaults'] = used
    return st


# ====================================================================== the conditions' levels
def _find(items, ident, key_uid='uid', key_name='name'):
    ident = str(ident or '')
    for x in items:
        if isinstance(x, dict) and ident in (str(x.get(key_uid) or ''), str(x.get(key_name) or '')):
            return x
        if not isinstance(x, dict) and ident in (str(getattr(x, key_uid, '') or ''), str(getattr(x, key_name, ''))):
            return x
    return None


def _traj_duration(t) -> float | None:
    t = t.get('t') if isinstance(t, dict) else None
    if not isinstance(t, list) or len(t) < 2:
        return None
    d = _num(t[-1], 0.0) - _num(t[0], 0.0)
    return d if d > 0 else None


def project_tasks(project) -> list:
    """The default tasks: one per coordinated motion of project.tasks.trajectories (trajectories of different robots
    over the same time span are one motion: a bimanual task), else one per grasp set, else one called "Task"."""
    trajs = [t for t in project.tasks.trajectories if isinstance(t, dict)]
    out = []
    used = set()
    for i, t in enumerate(trajs):
        if i in used:
            continue
        span = (_num((t.get('t') or [0])[0], 0.0), _num((t.get('t') or [0])[-1], 0.0))
        group = [t]
        for j in range(i + 1, len(trajs)):
            u = trajs[j]
            s2 = (_num((u.get('t') or [0])[0], 0.0), _num((u.get('t') or [0])[-1], 0.0))
            if j not in used and u.get('robot_uid') not in {g.get('robot_uid') for g in group} and \
                    abs(s2[0] - span[0]) < 1e-6 and abs(s2[1] - span[1]) < 1e-6:
                group.append(u)
                used.add(j)
        names = [str(g.get('name') or 'Trajectory') for g in group]
        name = names[0] if len(names) == 1 else (os.path.commonprefix(names).rstrip(' ,(-') or names[0])
        out.append({'name': name, 'trajectories': [str(g.get('uid') or '') for g in group],
                    'episode_s': _traj_duration(t),
                    'episode_source': ('the trajectory\'s duration' if len(group) == 1 else
                                       f'the duration of its {len(group)} trajectories (one motion of '
                                       f'{len(group)} robots)')})
    if out:
        return out
    for g in project.tasks.grasp_sets:
        if isinstance(g, dict):
            out.append({'name': str(g.get('name') or 'Grasp set'), 'trajectories': [], 'episode_s': None,
                        'episode_source': ''})
    return out or [{'name': 'Task', 'trajectories': [], 'episode_s': None, 'episode_source': ''}]


def _levels(project, st, notes) -> dict:
    """{axis: [level dict]} with each level's 'name' and what it stands for."""
    lv = {}
    # tasks
    if st['tasks']:
        tasks = []
        for x in st['tasks']:
            x = x if isinstance(x, dict) else {'name': str(x)}
            name = str(x.get('name') or f'Task {len(tasks) + 1}')
            tr = x.get('trajectory') or x.get('trajectories')
            trs = tr if isinstance(tr, list) else ([tr] if tr else [])
            found = [_find(project.tasks.trajectories, i) for i in trs]
            for i, f in zip(trs, found):
                if f is None:
                    raise CampaignError(f'The task "{name}" names a trajectory "{i}" the project does not have '
                                        '(Tasks ▸ Trajectories lists them).')
            e = _num(x.get('episode_s'))
            dur = max((d for d in (_traj_duration(f) for f in found) if d), default=None)
            tasks.append({'name': name, 'trajectories': [f.get('uid') for f in found],
                          'episode_s': e if e is not None else dur,
                          'episode_source': 'your figure' if e is not None else (
                              'the trajectory\'s duration' if dur else '')})
        lv['task'] = tasks
    else:
        lv['task'] = project_tasks(project)
    # variants
    vs = []
    for x in st['variants'] or [project.active_uid]:
        v = project.variant_by_uid(str(x)) or next((v for v in project.variants if v.name == str(x)), None)
        if v is None:
            raise CampaignError(f'The campaign names a rig variant "{x}" the project does not have (it has: '
                                f'{", ".join(v.name for v in project.variants)}).')
        vs.append({'name': v.name, 'uid': v.uid})
    lv['variant'] = vs
    # lighting presets
    lights = [x for x in getattr(project.workspace, 'lights', []) or [] if isinstance(x, dict)]
    pres = []
    for x in st['lighting'] or [{'name': 'As designed', 'lights': {}}]:
        x = x if isinstance(x, dict) else {'name': str(x)}
        name = str(x.get('name') or f'Lighting {len(pres) + 1}')
        dims = {}
        for ident, dim in (x.get('lights') or {}).items() if isinstance(x.get('lights'), dict) else ():
            li = _find(lights, ident)
            if li is None:
                raise CampaignError(f'The lighting preset "{name}" names a light "{ident}" the workspace does not '
                                    'have (Workspace ▸ Lights lists them).')
            v = _num(dim)
            if v is None:
                v = 1.0 if dim is True or str(dim).lower() == 'on' else 0.0
            dims[str(li.get('uid'))] = _clamp(v, 0.0, 1.0)
        pres.append({'name': name, 'lights': dims})
    lv['lighting'] = pres
    # object sets
    objs = list(project.workspace.objects)
    sets = []
    for x in st['object_sets'] or [{'name': 'As placed', 'objects': [o.uid for o in objs]}]:
        x = x if isinstance(x, dict) else {'name': str(x)}
        name = str(x.get('name') or f'Object set {len(sets) + 1}')
        uids = []
        for ident in x.get('objects') or []:
            o = _find(objs, ident)
            if o is None:
                raise CampaignError(f'The object set "{name}" names an object "{ident}" the workspace does not have.')
            uids.append(o.uid)
        sets.append({'name': name, 'objects': uids})
    lv['objects'] = sets
    # teleop devices
    devs = [d for d in project.tasks.teleop_devices if isinstance(d, dict)]
    dl = []
    for x in st['devices'] or [{'name': 'As in the project' if devs else 'None named',
                                'devices': [d.get('uid') for d in devs]}]:
        x = x if isinstance(x, dict) else {'name': str(x), 'devices': [x]}
        name = str(x.get('name') or f'Devices {len(dl) + 1}')
        uids = []
        for ident in x.get('devices') or []:
            d = _find(devs, ident)
            if d is None:
                raise CampaignError(f'The device level "{name}" names a teleop device "{ident}" the project does not '
                                    'have (Tasks ▸ Teleop devices lists them).')
            uids.append(str(d.get('uid')))
        dl.append({'name': name, 'devices': uids})
    lv['device'] = dl
    for ax in AXES:
        names = [x['name'] for x in lv[ax]]
        if len(set(names)) != len(names):
            raise CampaignError(f'Two {AXIS_TITLE[ax].lower()} levels share a name ({", ".join(names)}): give each '
                                'its own name.')
    return lv


def conditions(project, st, notes=None) -> tuple:
    """(levels, [condition]) : the full factorial of the levels, or the explicit list st['conditions']."""
    notes = notes if notes is not None else []
    lv = _levels(project, st, notes)
    rows = []
    if st['conditions']:
        for k, c in enumerate(st['conditions']):
            if not isinstance(c, dict):
                raise CampaignError(f'Condition {k + 1} is not an object such as {{"task": "Pick", "lighting": '
                                    '"Dim"}.')
            pick = {}
            for ax in AXES:
                want = c.get(ax)
                if want is None:
                    pick[ax] = 0
                    continue
                idx = next((i for i, x in enumerate(lv[ax]) if x['name'] == str(want) or x.get('uid') == want), None)
                if idx is None:
                    raise CampaignError(f'Condition {k + 1} names a {AXIS_TITLE[ax].lower()} "{want}" that is not '
                                        f'among the levels ({", ".join(x["name"] for x in lv[ax])}).')
                pick[ax] = idx
            rows.append(pick)
    else:
        n = math.prod(len(lv[ax]) for ax in AXES)
        if n > MAX_CONDITIONS:
            raise CampaignError(f'The levels make {n:,} conditions, more than {MAX_CONDITIONS}: list the conditions '
                                'you want (conditions) instead of every combination.')
        for combo in itertools.product(*[range(len(lv[ax])) for ax in AXES]):
            rows.append(dict(zip(AXES, combo)))
    if len(rows) > MAX_CONDITIONS:
        raise CampaignError(f'The campaign has {len(rows):,} conditions, more than {MAX_CONDITIONS}.')
    varying = [ax for ax in AXES if len({r[ax] for r in rows}) > 1]
    out = []
    for i, r in enumerate(rows):
        parts = [lv[ax][r[ax]]['name'] for ax in varying] or [lv['task'][r['task']]['name']]
        out.append({'id': f'C{i + 1}', 'label': ' · '.join(parts), 'levels': {ax: r[ax] for ax in AXES},
                    **{ax: lv[ax][r[ax]]['name'] for ax in AXES}, 'variant_uid': lv['variant'][r['variant']]['uid']})
    labels = [c['label'] for c in out]
    if len(set(labels)) != len(labels):
        for c in out:
            c['label'] = f'{c["id"]} {c["label"]}'
    return lv, out


# ====================================================================== the design
def williams(n: int) -> list:
    """The rows of a Williams balanced Latin square of order n (Williams 1949): the first row 0, 1, n−1, 2, n−2, …,
    the others its cyclic shifts; for odd n those n rows and their mirror images (2n rows)."""
    n = int(n)
    if n < 1:
        return []
    first, lo, hi = [0], 1, n - 1
    take_lo = True
    while len(first) < n:
        if take_lo:
            first.append(lo)
            lo += 1
        else:
            first.append(hi)
            hi -= 1
        take_lo = not take_lo
    rows = [[(x + i) % n for x in first] for i in range(n)]
    if n % 2 == 1 and n > 1:
        rows += [r[::-1] for r in rows]
    return rows


def choose_rounds(C: int, O: int, m0: int, wanted) -> tuple:
    """(B, position_exact, carryover_exact): rounds per operator (module docstring)."""
    L = len(williams(C))
    if isinstance(wanted, int):
        B = max(1, min(wanted, max(1, m0)))
    else:
        cand = range(1, max(1, min(m0, MAX_ROUNDS)) + 1)
        B = next((b for b in cand if (O * b) % L == 0), None) or next((b for b in cand if (O * b) % C == 0), None) or 1
    S = O * B
    return B, S % C == 0, (S % L == 0) if C > 1 else True


def design(C: int, O: int, B: int, seed: int) -> dict:
    """The sequences: {'rows': the square, 'symbol_map': condition index per symbol, 'sequences': [{'operator',
    'round', 'row', 'order': [condition index]}]} for S = O × B sequences (module docstring)."""
    rows = williams(C)
    L = len(rows)
    rng = np.random.default_rng([int(seed), 0x5EED, C, O, B])
    sym = [int(x) for x in rng.permutation(C)]
    take = [s % L for s in range(O * B)]
    order = [int(x) for x in rng.permutation(len(take))]
    seqs = []
    for slot, k in enumerate(order):
        o, r = divmod(slot, B)
        row = take[k]
        seqs.append({'operator': o, 'round': r, 'row': row, 'order': [sym[x] for x in rows[row]]})
    seqs.sort(key=lambda s: (s['operator'], s['round']))
    return {'rows': rows, 'symbol_map': sym, 'sequences': seqs}


def power_n(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8, continuity: bool = True) -> dict:
    """Demonstrations per condition to tell success rate p1 from p2 (module docstring): {'n' (rounded up), 'n_exact',
    'method', 'z_alpha', 'z_beta'}, through core/stats.py's n_two_proportions when it is there (then also
    'power_at_n' and 'formula_n', this module's own figure)."""
    p1, p2 = float(p1), float(p2)
    if not (0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0) or abs(p2 - p1) < 1e-9:
        raise CampaignError('The power analysis needs two different success rates between 0 and 1 (p_base and '
                            'p_target), e.g. 0.5 and 0.7.')
    nd = NormalDist()
    za, zb = nd.inv_cdf(1.0 - alpha / 2.0), nd.inv_cdf(power)
    pb = (p1 + p2) / 2.0
    d = abs(p2 - p1)
    n = (za * math.sqrt(2.0 * pb * (1.0 - pb)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / d ** 2
    method = 'normal approximation (Fleiss, Levin & Paik 2003)'
    if continuity:
        n = n / 4.0 * (1.0 + math.sqrt(1.0 + 4.0 / (n * d))) ** 2
        method += ' with the continuity correction (Fleiss, Tytun & Ury 1980)'
    out = {'n': int(math.ceil(n - 1e-9)), 'n_exact': n, 'method': method, 'z_alpha': za, 'z_beta': zb}
    fn = _stats_function()
    if fn is not None:
        try:
            r = fn(p1, p2, alpha=alpha, power=power, method='normal', continuity=continuity)
            k = int(r['n_per_group'])
        except Exception:                               # noqa: BLE001 — refused or another shape: the formula here
            return out
        out.update(n=k, method=f'core.stats.{STATS_FUNCTION}: {method}', power_at_n=_num(r.get('power_at_n')),
                   formula_n=out['n'])
    return out


def _stats_function():
    try:
        from . import stats as ST
    except ImportError:
        return None
    fn = getattr(ST, STATS_FUNCTION, None)
    return fn if callable(fn) else None


# ====================================================================== storage
def storage(project, variant_uid: str, st: dict) -> dict:
    """GB per second of recording of a variant (module docstring): {'gb_per_s', 'payload_mbps', 'ratio',
    'ratio_source', 'container_overhead', 'sensors': [{'uid', 'name', 'payload_mbps', 'rate_source'}],
    'incomplete': [names whose rate is not stated]}."""
    from .studies import bandwidth as BW
    snap = project if project.active_uid == variant_uid else _with_variant(project, variant_uid)
    bst = BW.settings_of(snap)
    res = BW.budget(snap, bst)
    rows = [{'uid': r['uid'], 'name': r['name'], 'payload_mbps': r['payload_mbps'], 'rate_source': r['rate_source']}
            for r in res['sensors']]
    payload = sum(r['payload_mbps'] for r in rows if r['payload_mbps'] is not None)
    fmt = st['format']
    cr = st.get('compression_ratio')
    if fmt == 'lerobot' and cr:
        ratio, src = cr['value'], 'your measured ratio' + (f': {cr["source"]}' if cr['source'] else ' (no source '
                                                                                                    'given)')
    elif fmt == 'lerobot':
        from .datasets import lerobot as LR
        e = LR.ENCODER_DEFAULTS
        ratio, src = 1.0, (f'raw size: LeRobot encodes video (AV1 with libsvtav1, {e["pix_fmt"]}, g {e["g"]}, crf '
                           f'{e["crf"]}), but no source states the size that gives (it depends on the content); '
                           'enter a ratio measured on a recorded episode')
    else:
        ratio, src = 1.0, 'raw: the frames as streamed'
    ov = float(bst['container_overhead'])
    mbps = payload * ratio * (1.0 + ov)
    return {'gb_per_s': mbps / 8.0 / 1000.0, 'payload_mbps': payload, 'recorded_mbps': mbps, 'ratio': ratio,
            'ratio_source': src, 'container_overhead': ov,
            'container_overhead_default': 'container_overhead' in bst.get('defaults', []),
            'sensors': rows, 'incomplete': [r['name'] for r in rows if r['payload_mbps'] is None]}


def _with_variant(project, uid):
    p = project.snapshot()
    p.active_uid = uid
    return p


# ====================================================================== the checklist
def _exposures(project, variant_uid) -> list:
    """Per enabled camera of the variant: its exposure, gain and polariser as the motion-blur and lighting studies'
    settings hold them (the figures their Apply writes), or "not set"."""
    v = project.variant_by_uid(variant_uid) or project.variant
    mb = (project.analysis.studies.get('motion_blur') or {}).get('exposure_s') or {}
    lt = (project.analysis.studies.get('lighting') or {}).get('sensors') or {}
    out = []
    for s in v.sensors:
        if not s.enabled or s.kind not in ('camera', 'depth'):
            continue
        e = mb.get(s.uid) if isinstance(mb, dict) else None
        cfg = lt.get(s.uid) if isinstance(lt, dict) and isinstance(lt.get(s.uid), dict) else {}
        exp = src = None
        if isinstance(e, dict) and _num(e.get('value')):
            exp, src = float(e['value']), 'motion blur: ' + (str(e.get('source') or '') or 'your figure')
        elif _num(cfg.get('exposure_s')):
            exp, src = float(cfg['exposure_s']), 'lighting: your setting'
        gain = _num(cfg.get('gain'))
        out.append({'sensor_uid': s.uid, 'sensor': s.name, 'exposure_s': exp, 'source': src, 'gain': gain,
                    'polariser': bool(cfg.get('polariser'))})
    return out


def _exposure_text(e) -> str:
    if e['exposure_s'] is None:
        t = f'{e["sensor"]}: exposure not set (run Analysis ▸ Motion blur and Apply its exposure, then set it on the ' \
            'camera)'
    else:
        t = f'{e["sensor"]}: exposure {e["exposure_s"] * 1000:.3g} ms ({e["source"]})'
    if e['gain'] is not None:
        t += f', gain {e["gain"]:g}'
    if e['polariser']:
        t += ', linear polariser fitted'
    return t


def _lighting_text(project, preset) -> str:
    lights = [x for x in getattr(project.workspace, 'lights', []) or [] if isinstance(x, dict)]
    if not lights:
        return f'Lighting "{preset["name"]}": the workspace has no lights in the project (room light as it is).'
    parts = []
    for li in lights:
        dim = preset['lights'].get(str(li.get('uid')))
        if dim is None:
            on = li.get('enabled', True)
            dim = _num(li.get('dim'), 1.0) if on else 0.0
        parts.append(f'{li.get("name") or "light"} ' + ('off' if dim <= 0 else f'on at {dim * 100:.0f} %'))
    return f'Lighting "{preset["name"]}": ' + ', '.join(parts) + '.'


def _calibration_text(project, st) -> str:
    from .calib import planner as PL
    from .calib import records as RC
    hp = project.analysis.studies.get('handeye_plan') or {}
    ps = PL.PlanSettings.from_dict(hp)
    cam = None
    if hp.get('sensor'):
        cam = project.variant.sensor(str(hp['sensor']))
    if cam is None:
        cams = [s for s in project.variant.sensors if s.enabled and s.kind in ('camera', 'depth')]
        cam = next((s for s in cams if s.mount.on_link), cams[0] if cams else None)
    boards = project.calibration.get('boards') or []
    board = _find(boards, hp.get('board')) if hp.get('board') else (boards[0] if boards else None)
    if board is not None:
        btxt = f'board "{board.get("name") or "board"}"'
    else:
        b = RC.DEFAULT_BOARD
        btxt = (f'DexVantage\'s default board (ChArUco {b["squares_x"]} × {b["squares_y"]}, '
                f'{b["square_m"] * 1000:g} mm squares, a default)')
    sessions = [s for s in project.calibration.get('sessions') or [] if isinstance(s, dict) and s.get('result')]
    last = f' Last solved session: "{sessions[-1].get("name") or "session"}".' if sessions else ''
    need = ('' if 'need_mm' in hp else ' (defaults)')
    return (f'Calibration check: capture {st["calibration"]["check_poses"]} poses of the hand-eye plan '
            f'(Calibrate ▸ Hand-eye pose plan, {ps.n_poses} planned) with {btxt}'
            + (f' on {cam.name}' if cam is not None else '')
            + f', solve, and confirm the mount is within {ps.need_mm:g} mm and {ps.need_deg:g}°{need} of the plan; '
            'recalibrate if not.' + last)


# ====================================================================== the plan
def plan(project, spec: dict | None = None, ctx=None) -> dict:
    """The campaign plan (module docstring)."""
    report = (lambda f, t='': ctx.report(f, t)) if ctx is not None else (lambda f, t='': None)
    st = settings_of(project, spec)
    notes: list = []
    cr = st.get('compression_ratio')
    if st['format'] == 'lerobot' and cr and cr['value'] > 1.0:
        notes.append(f'The compression ratio is {cr["value"]:g}: it is the encoded size over the raw size, so above 1 '
                     'the recording is larger than raw (values above 10 are taken as 10). For a 20:1 encoder enter '
                     '0.05.')
    report(0.05, 'Conditions')
    lv, conds = conditions(project, st, notes)
    C, O = len(conds), len(st['operators'])
    # ---- demonstrations per condition
    dm = st['demos']
    pw = None
    if dm['mode'] == 'power':
        pw = power_n(dm['p_base'], dm['p_target'], dm['alpha'], dm['power'], dm['continuity'])
        target = pw['n']
    else:
        target = dm['per_condition']
    keep = st['keep_rate']
    need = int(math.ceil(target / keep - 1e-9))
    m0 = int(math.ceil(need / O))
    B, pos_exact, carry_exact = choose_rounds(C, O, m0, st['rounds'])
    m = int(math.ceil(m0 / B) * B)
    k = m // B
    per_cond = O * m
    if per_cond * C > MAX_EPISODES:
        raise CampaignError(f'The campaign would record {per_cond * C:,} episodes, more than {MAX_EPISODES:,}: lower '
                            'the demonstrations per condition or the number of conditions.')
    if per_cond > need:
        notes.append(f'Each condition records {per_cond:,} episodes, {per_cond - need:,} more than the {need:,} '
                     f'needed, so that each of the {O} operator{"s" if O > 1 else ""} records it equally often in '
                     f'{B} block{"s" if B > 1 else ""} of {k} episode{"s" if k > 1 else ""}.')
    report(0.2, 'Storage')
    # ---- per condition: time and storage
    stor = {}
    for v in lv['variant']:
        stor[v['uid']] = storage(project, v['uid'], st)
    task_by_name = {t['name']: t for t in lv['task']}
    for c in conds:
        t = task_by_name[c['task']]
        if st['episode_s'] is not None:
            ep, src = st['episode_s'], 'your figure (every task)'
        elif t.get('episode_s'):
            ep, src = float(t['episode_s']), t.get('episode_source') or 'your figure'
        else:
            ep, src = DEFAULT_EPISODE_S, 'a default (the task has no trajectory: enter its duration)'
        s = stor[c['variant_uid']]
        c.update({'episode_s': ep, 'episode_source': src, 'episode_default': src.startswith('a default'),
                  'gb_per_episode': s['gb_per_s'] * ep, 'storage_incomplete': bool(s['incomplete']),
                  'target': target, 'recorded': per_cond, 'expected_kept': per_cond * keep})
    # ---- the design
    report(0.4, 'Design')
    dz = design(C, O, B, st['seed'])
    L = len(dz['rows'])
    # ---- sessions per operator
    T = int(st['session_max_min'])
    reset = st['reset_s']
    chg = st['changeover_min'] * 60.0
    cal_session = st['calibration']['every'] == 'session'
    cal_s = st['calibration']['minutes'] * 60.0
    def pack(episodes, o, cap):
        """Episodes in order into sessions of at most cap minutes; None when one episode does not fit."""
        sessions, cur, extra = [], None, 0.0
        for ci, rnd in episodes:
            dur = conds[ci]['episode_s'] + reset
            if cur is not None:
                extra = 0.0 if cur['blocks'][-1]['ci'] == ci else chg
                if math.ceil((cur['seconds'] + extra + dur) / 60.0 - 1e-9) > cap:
                    sessions.append(cur)
                    cur = None
            if cur is None:
                base = chg + (cal_s if cal_session else 0.0)
                if math.ceil((base + dur) / 60.0 - 1e-9) > cap:
                    return None
                cur = {'operator': o, 'seconds': base, 'blocks': [], 'calibration': cal_session}
                extra = 0.0
            if not cur['blocks'] or cur['blocks'][-1]['ci'] != ci or cur['blocks'][-1]['round'] != rnd:
                cur['blocks'].append({'ci': ci, 'round': rnd, 'episodes': 0, 'seconds': 0.0})
            cur['seconds'] += extra + dur
            cur['blocks'][-1]['episodes'] += 1
            cur['blocks'][-1]['seconds'] += dur
        if cur is not None:
            sessions.append(cur)
        return sessions

    op_sessions = []
    for o in range(O):
        seq = [s for s in dz['sequences'] if s['operator'] == o]
        episodes = [(ci, s['round']) for s in seq for ci in s['order'] for _ in range(k)]
        sessions = pack(episodes, o, T)
        if sessions is None:
            c = max(conds, key=lambda x: x['episode_s'])
            raise CampaignError(f'One episode of {c["label"]} ({c["episode_s"]:g} s + {reset:g} s reset, with '
                                f'{st["changeover_min"]:g} min to set it up'
                                + (f' and {st["calibration"]["minutes"]:g} min of calibration' if cal_session else '')
                                + f') does not fit in a {T} min session: raise the session limit or shorten the '
                                'episode.')
        # even the sessions out: the shortest cap that still needs no more sessions (greedy packing needs no more
        # sessions as the cap grows, so the cap is found by bisection), so no session is a short remainder
        lo, hi = 1, T
        while lo < hi:
            mid = (lo + hi) // 2
            trial = pack(episodes, o, mid)
            if trial is not None and len(trial) <= len(sessions):
                hi = mid
            else:
                lo = mid + 1
        op_sessions.append(pack(episodes, o, lo))
    # ---- days
    report(0.6, 'Days')
    d0 = parse_hhmm(st['day_start'])
    day_end = d0 + int(round(st['day_hours'] * 60))
    cal_day = (0 if cal_session else int(math.ceil(cal_s / 60.0 - 1e-9)))
    if d0 + cal_day + min((math.ceil(s['seconds'] / 60.0 - 1e-9) for ss in op_sessions for s in ss), default=0) \
            > day_end:
        raise CampaignError(f'No session fits in a {st["day_hours"]:g} h day after the calibration check: lengthen the '
                            'day or shorten the sessions.')
    queues = [list(ss) for ss in op_sessions]
    placed = []
    day = 0
    start_date = parse_date(st['start_date']) if st['start_date'] else None
    dates = _dates(start_date, st['weekdays_only'])
    while any(queues):
        day += 1
        if day > 20000:
            raise CampaignError('The campaign does not end within 20 000 days: check the session and day limits.')
        t = d0 + cal_day
        n_today, per_op = 0, [0] * O
        today = []
        progress = True
        while progress and n_today < st['sessions_per_day']:
            progress = False
            for i in range(O):
                o = (i + day - 1) % O
                if not queues[o] or per_op[o] >= st['operator_sessions_per_day'] or \
                        n_today >= st['sessions_per_day']:
                    continue
                s = queues[o][0]
                mins = int(math.ceil(s['seconds'] / 60.0 - 1e-9))
                if t + mins > day_end:
                    continue
                queues[o].pop(0)
                today.append({**s, 'day': day, 'slot': n_today + 1, 'start_min': t, 'end_min': t + mins,
                              'minutes': mins})
                t += mins + st['break_min']
                n_today += 1
                per_op[o] += 1
                progress = True
        if not today:
            raise CampaignError('A session does not fit in a day: lengthen the day (day_hours) or lower the session '
                                'limit.')
        if not cal_session and today:
            today[0]['calibration'] = True
        placed += today
    # ---- the sessions as written
    report(0.8, 'Checklists')
    expo = {v['uid']: _exposures(project, v['uid']) for v in lv['variant']}
    cal_text = _calibration_text(project, st)
    sessions = []
    first_ep = [[0] * C for _ in range(O)]
    for i, s in enumerate(placed):
        op = st['operators'][s['operator']]
        blocks, gb = [], 0.0
        for b in s['blocks']:
            c = conds[b['ci']]
            n0 = first_ep[s['operator']][b['ci']]
            first_ep[s['operator']][b['ci']] += b['episodes']
            g = c['gb_per_episode'] * b['episodes']
            gb += g
            blocks.append({'condition': c['id'], 'label': c['label'], 'round': b['round'] + 1,
                           'episodes': b['episodes'], 'first': n0 + 1, 'minutes': _r(b['seconds'] / 60.0, 2),
                           'gb': _r(g, 4)})
        date = dates(s['day'])
        sess = {'index': i + 1, 'id': f'S{i + 1}', 'day': s['day'], 'date': date.isoformat() if date else None,
                'slot': s['slot'], 'start': _hhmm(s['start_min']), 'end': _hhmm(s['end_min']),
                'start_min': s['start_min'], 'end_min': s['end_min'], 'minutes': s['minutes'],
                'operator': op, 'operator_index': s['operator'], 'calibration': bool(s['calibration']),
                'episodes': sum(b['episodes'] for b in blocks), 'gb': _r(gb, 4), 'blocks': blocks}
        sess['checklist'] = _checklist(project, st, lv, conds, sess, expo, cal_text)
        sessions.append(sess)
    # ---- totals, days, budget
    days = []
    for d in range(1, day + 1):
        ss = [s for s in sessions if s['day'] == d]
        date = dates(d)
        days.append({'day': d, 'date': date.isoformat() if date else None, 'sessions': len(ss),
                     'episodes': sum(s['episodes'] for s in ss), 'gb': _r(sum(s['gb'] for s in ss), 4),
                     'start': _hhmm(d0), 'end': _hhmm(max(s['end_min'] for s in ss)) if ss else _hhmm(d0),
                     'operators': sorted({s['operator'] for s in ss}, key=st['operators'].index)})
    total_ep = sum(s['episodes'] for s in sessions)
    total_gb = sum(c['gb_per_episode'] * c['recorded'] for c in conds)
    op_hours = {op: _r(sum(s['minutes'] for s in sessions if s['operator'] == op) / 60.0, 3) for op in
                st['operators']}
    incomplete = sorted({n for v in stor.values() for n in v['incomplete']})
    totals = {'conditions': C, 'operators': O, 'episodes': total_ep, 'expected_kept': _r(total_ep * keep, 2),
              'days': day, 'calendar_days': (dates(day) - dates(1)).days + 1 if start_date else None,
              'sessions': len(sessions), 'rig_hours': _r(sum(s['minutes'] for s in sessions) / 60.0, 3),
              'recording_hours': _r(sum(c['episode_s'] * c['recorded'] for c in conds) / 3600.0, 3),
              'operator_hours': op_hours, 'storage_gb': _r(total_gb, 3),
              'storage_bound': 'at_least' if incomplete else None, 'storage_incomplete': incomplete,
              'gb_per_day_max': _r(max((d['gb'] for d in days), default=0.0), 3),
              'first_date': days[0]['date'] if days else None, 'last_date': days[-1]['date'] if days else None}
    budget = _budget(st, totals, notes)
    if total_ep != per_cond * C:                      # the packing lost or duplicated an episode: a bug
        raise RuntimeError(f'campaign: {total_ep} episodes scheduled against {per_cond * C} planned')
    bal_text = _design_text(C, O, B, k, L, pos_exact, carry_exact, st['seed'])
    head_status, head_word = budget['status'], budget['word']
    if incomplete and head_status == 'good':
        head_status, head_word = 'warning', 'storage incomplete'
    storage_txt = f'{total_gb:,.1f} GB' if not incomplete else f'at least {total_gb:,.1f} GB'
    text = f'{total_ep:,} episodes over {_plural(day, "day")}, {storage_txt}'
    explain = (f'{C} condition{"s" if C != 1 else ""} × {_plural(per_cond, "episode")} ({_plural(O, "operator")}, '
               f'{_plural(len(sessions), "session")}, {totals["rig_hours"]:.1f} h on the rig). ' + budget['text'])
    doc = {
        'name': st['name'], 'settings': {k2: v for k2, v in st.items()},
        'units': 'times in minutes after midnight (local, no time zone) and hours; storage in GB = 1e9 bytes; data '
                 'rates in Mbit/s = 1e6 bit/s',
        'levels': lv, 'conditions': conds, 'operators': list(st['operators']),
        'demos': {'mode': dm['mode'], 'target': target, 'keep_rate': keep, 'needed': need, 'recorded': per_cond,
                  'per_operator': m, 'power': pw},
        'design': {'kind': 'Williams balanced Latin square', 'conditions': C, 'operators': O, 'rounds': B,
                   'block_episodes': k, 'square_rows': L, 'sequences': O * B, 'seed': st['seed'],
                   'position_balance': 'exact' if pos_exact else 'partial',
                   'carryover_balance': 'exact' if carry_exact else 'partial', 'symbol_map': dz['symbol_map'],
                   'text': bal_text},
        'sequences': [{'operator': st['operators'][s['operator']], 'round': s['round'] + 1, 'row': s['row'],
                       'order': [conds[i]['id'] for i in s['order']]} for s in dz['sequences']],
        'storage': {v['name']: {kk: (_r(vv, 9) if isinstance(vv, float) else vv) for kk, vv in stor[v['uid']].items()}
                    for v in lv['variant']},
        'sessions': sessions, 'days': days, 'totals': totals, 'budget': budget,
        'headline': {'text': text, 'episodes': total_ep, 'days': day, 'storage_gb': _r(total_gb, 3),
                     'storage_bound': totals['storage_bound'], 'status': head_status, 'status_word': head_word,
                     'explain': explain},
        'advice': budget['advice'], 'notes': notes, 'assumptions': list(ASSUMPTIONS), 'references': list(REFERENCES),
        'defaults': list(st['defaults']),
    }
    report(1.0, '')
    return doc


def _dates(start: _dt.date | None, weekdays_only: bool):
    """day number (1-based) -> its date (None without a start date); weekends skipped when weekdays_only."""
    seq: list = []                                   # seq[i] = the date of day i + 1

    def at(n):
        if start is None:
            return None
        if not weekdays_only:
            return start + _dt.timedelta(days=n - 1)
        while len(seq) < n:
            d = seq[-1] + _dt.timedelta(days=1) if seq else start
            while d.weekday() >= 5:
                d += _dt.timedelta(days=1)
            seq.append(d)
        return seq[n - 1]
    return at


def _checklist(project, st, lv, conds, sess, expo, cal_text) -> list:
    """The session's checklist: the calibration check (when the session carries it), then per item kind (rig,
    lighting, objects, device, exposure) one line when every condition of the session has the same, else one line
    per condition prefixed by it; the disk to free; the blocks to record."""
    out = []
    if sess['calibration']:
        out.append({'item': 'calibration', 'text': cal_text})
    ids = list(dict.fromkeys(b['condition'] for b in sess['blocks']))
    per: dict = {}                                   # item -> [(condition, key, text)]
    for cid in ids:
        c = next(x for x in conds if x['id'] == cid)
        lev = c['levels']
        per.setdefault('rig', []).append((c, '', f'Rig variant "{c["variant"]}".'))
        per.setdefault('lighting', []).append((c, '', _lighting_text(project, lv['lighting'][lev['lighting']])))
        objs = lv['objects'][lev['objects']]
        names = [o.name for o in project.workspace.objects if o.uid in objs['objects']]
        per.setdefault('objects', []).append((c, '', f'Objects "{objs["name"]}": ' + (
            ', '.join(names) if names else 'none in the project') + '.'))
        dv = lv['device'][lev['device']]
        dnames = [str(d.get('name') or 'device') for d in project.tasks.teleop_devices
                  if isinstance(d, dict) and str(d.get('uid')) in dv['devices']]
        per.setdefault('device', []).append((c, '', f'Teleop "{dv["name"]}": ' + (
            '; '.join(dnames) if dnames else 'no device named in the project') + '.'))
        for e in expo.get(c['variant_uid'], []):
            per.setdefault('exposure', []).append((c, e['sensor_uid'], _exposure_text(e) + '.'))
    for item in ('rig', 'lighting', 'objects', 'device', 'exposure'):
        rows = per.get(item, [])
        texts = {}
        for c, key, t in rows:
            texts.setdefault(key, set()).add(t)
        same = all(len(v) == 1 for v in texts.values())
        done = set()
        for c, key, t in rows:
            if same and (key, t) in done:
                continue
            done.add((key, t))
            row = {'item': item, 'text': t if same else f'{c["id"]} ({c["label"]}): {t}'}
            if not same:
                row['condition'] = c['id']
            if key:
                row['sensor_uid'] = key
            out.append(row)
    out.append({'item': 'disk', 'text': f'Free at least {sess["gb"]:,.2f} GB on the recorder before the session'
                + (' (at least: a sensor\'s data rate is not stated).' if any(
                    next(x for x in conds if x['id'] == b['condition'])['storage_incomplete'] for b in sess['blocks'])
                   else '.')})
    for b in sess['blocks']:
        out.append({'item': 'episodes', 'condition': b['condition'],
                    'text': f'Record {b["condition"]} ({b["label"]}): episodes {b["first"]}–'
                            f'{b["first"] + b["episodes"] - 1} of this operator ({b["episodes"]}).'})
    return out


def _budget(st, totals, notes) -> dict:
    b = st['budget']
    checks = []

    def add(key, value, limit, unit, what):
        share = value / limit if limit > 0 else math.inf
        status = 'critical' if share > 1.0 + 1e-9 else ('warning' if share > TIGHT else 'good')
        checks.append({'key': key, 'what': what, 'value': _r(value, 3), 'limit': limit, 'unit': unit,
                       'share': _r(share, 4), 'status': status})
    if 'disk_gb' in b:
        add('disk_gb', totals['storage_gb'], b['disk_gb'], 'GB', 'Storage')
    if 'days' in b:
        add('days', totals['days'], b['days'], 'days', 'Days')
    if 'operator_hours' in b:
        add('operator_hours', max(totals['operator_hours'].values()), b['operator_hours'], 'h',
            'Recording per operator (the busiest)')
    up_h = None
    if 'upload_mbps' in b:
        up_h = totals['gb_per_day_max'] * 8000.0 / b['upload_mbps'] / 3600.0
        totals['upload_hours_busiest_day'] = _r(up_h, 3)
        totals['upload_hours_total'] = _r(totals['storage_gb'] * 8000.0 / b['upload_mbps'] / 3600.0, 3)
        lim = b.get('upload_hours_per_day', 24.0)
        add('upload_hours_per_day', up_h, lim, 'h', 'Upload of the busiest day' +
            ('' if 'upload_hours_per_day' in b else ' (against 24 h: no upload window set)'))
    elif 'upload_hours_per_day' in b:
        notes.append('The upload window is set without the upload link\'s rate (upload_mbps): it is not checked.')
    rank = {'good': 0, 'warning': 1, 'critical': 2}
    advice = []
    if not checks:
        return {'checks': [], 'status': 'warning', 'word': 'no budget set', 'advice': advice,
                'text': 'No budget is set: enter the disk, the days or the upload link to judge the plan.'}
    worst = max(checks, key=lambda c: rank[c['status']])['status']
    for c in checks:
        if c['status'] == 'good':
            continue
        if c['key'] == 'disk_gb':
            advice.append(f'Storage {c["value"]:,.1f} GB against {c["limit"]:,.0f} GB: record to a compressed format '
                          '(enter its measured ratio), lower a frame rate on the Bandwidth page, or add disk.')
        elif c['key'] == 'days':
            advice.append(f'{c["value"]:.0f} days against {c["limit"]:g}: add an operator, allow more sessions a day, '
                          'or record fewer demonstrations per condition.')
        elif c['key'] == 'operator_hours':
            advice.append(f'The busiest operator records {c["value"]:.1f} h against {c["limit"]:g} h: add an '
                          'operator.')
        elif c['key'] == 'upload_hours_per_day':
            advice.append(f'The busiest day takes {c["value"]:.1f} h to upload against {c["limit"]:g} h: a faster '
                          'link, compression, or carry the disk.')
    over = [c for c in checks if c['status'] == 'critical']
    text = ('Over budget: ' + ', '.join(f'{c["what"].lower()} {c["share"] * 100:.0f} %' for c in over) + '.' if over
            else 'Within budget: ' + ', '.join(f'{c["what"].lower()} {c["share"] * 100:.0f} %' for c in checks) + '.')
    return {'checks': checks, 'status': worst, 'word': WORDS[worst], 'advice': advice, 'text': text}


def _design_text(C, O, B, k, L, pos, carry, seed) -> str:
    S = O * B
    t = (f'Williams balanced Latin square of order {C} ({L} rows): {S} sequence{"s" if S != 1 else ""} '
         f'({O} operator{"s" if O != 1 else ""} × {B} round{"s" if B != 1 else ""}), each a block of {k} episode'
         f'{"s" if k != 1 else ""} per condition; conditions and rows shuffled with seed {seed}. Each condition is '
         f'recorded {B} time{"s" if B != 1 else ""} by every operator')
    if C == 1:
        return t + '.'
    t += '; at every position equally often' if pos else (f'; at every position equally often only with a multiple '
                                                          f'of {C} sequences (now {S}: partial)')
    t += ('; and after every other condition equally often (carry-over balanced).' if carry else
          f'; carry-over balanced only with a multiple of {L} sequences (now {S}: partial).')
    return t


# ====================================================================== balance (the checks and the tests)
def balance(doc) -> dict:
    """Counts read off the plan's sessions: per operator × condition (episodes), per round position × condition
    (blocks), per ordered pair of adjacent conditions within a sequence; and whether each is equal."""
    conds = [c['id'] for c in doc['conditions']]
    ops = doc['operators']
    per_op = {o: {c: 0 for c in conds} for o in ops}
    for s in doc['sessions']:
        for b in s['blocks']:
            per_op[s['operator']][b['condition']] += b['episodes']
    pos = {p: {c: 0 for c in conds} for p in range(len(conds))}
    pairs = {}
    for q in doc['sequences']:
        for p, c in enumerate(q['order']):
            pos[p][c] += 1
        for a, b in zip(q['order'], q['order'][1:]):
            pairs[(a, b)] = pairs.get((a, b), 0) + 1
    allp = [pairs.get((a, b), 0) for a in conds for b in conds if a != b]
    return {'per_operator': per_op,
            'operator_equal': all(len(set(v.values())) == 1 for v in per_op.values()) and
            len({x for v in per_op.values() for x in v.values()}) == 1,
            'per_position': pos, 'position_equal': all(len(set(v.values())) == 1 for v in pos.values()),
            'pairs': {f'{a}>{b}': n for (a, b), n in sorted(pairs.items())},
            'carryover_equal': len(set(allp)) <= 1}


# ====================================================================== exports
CSV_COLUMNS = ('session', 'day', 'date', 'start', 'end', 'operator', 'calibration', 'condition', 'label', 'task',
               'variant', 'lighting', 'objects', 'device', 'round', 'first_episode', 'episodes', 'minutes', 'gb')


def to_csv(doc) -> str:
    """One row per block of every session (CSV_COLUMNS)."""
    by = {c['id']: c for c in doc['conditions']}
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(CSV_COLUMNS)
    for s in doc['sessions']:
        for b in s['blocks']:
            c = by[b['condition']]
            w.writerow([s['id'], s['day'], s['date'] or '', s['start'], s['end'], s['operator'],
                        'yes' if s['calibration'] else 'no', c['id'], c['label'], c['task'], c['variant'],
                        c['lighting'], c['objects'], c['device'], b['round'], b['first'], b['episodes'],
                        b['minutes'], b['gb']])
    return buf.getvalue()


def _ics_text(s: str) -> str:
    """RFC 5545 §3.3.11 TEXT: backslash, semicolon and comma escaped, line breaks as \\n."""
    return (str(s).replace('\\', '\\\\').replace(';', '\\;').replace(',', '\\,').replace('\r\n', '\n')
            .replace('\r', '\n').replace('\n', '\\n'))


def _fold(line: str) -> str:
    """RFC 5545 §3.1: lines of at most 75 octets, continued after CRLF and one space; never inside a UTF-8 sequence."""
    b = line.encode('utf-8')
    if len(b) <= 75:
        return line + '\r\n'
    out, cur, limit = [], b'', 75
    for ch in line:
        e = ch.encode('utf-8')
        if len(cur) + len(e) > limit:
            out.append(cur)
            cur, limit = b'', 74                       # the leading space takes one octet
        cur += e
    out.append(cur)
    return '\r\n '.join(x.decode('utf-8') for x in out) + '\r\n'


def plan_digest(doc) -> str:
    """A short digest of the plan's schedule (the calendar's UIDs are built from it)."""
    key = [(s['day'], s['start'], s['end'], s['operator'], [(b['condition'], b['episodes']) for b in s['blocks']])
           for s in doc['sessions']]
    return hashlib.sha256(json.dumps([doc.get('name'), key], sort_keys=True).encode('utf-8')).hexdigest()[:16]


def session_dates(doc, start_date=None) -> dict:
    """session index -> date: the plan's dates, else days counted from start_date (weekdays only when the plan says
    so), else from today (the export's date)."""
    sd = parse_date(start_date) if start_date else None
    if sd is None and all(s['date'] for s in doc['sessions']):
        return {s['index']: parse_date(s['date']) for s in doc['sessions']}
    at = _dates(sd or _dt.date.today(), bool(doc['settings'].get('weekdays_only', True)))
    return {s['index']: at(s['day']) for s in doc['sessions']}


def ics_events(doc, start_date=None) -> list:
    """The calendar's events (what to_ics writes and what a parser must read back): [{'uid', 'start', 'end' (naive
    datetimes, floating local time), 'summary', 'description'}]."""
    dates = session_dates(doc, start_date)
    dg = plan_digest(doc)
    out = []
    for s in doc['sessions']:
        d = dates[s['index']]
        t0 = _dt.datetime.combine(d, _dt.time()) + _dt.timedelta(minutes=s['start_min'])
        t1 = _dt.datetime.combine(d, _dt.time()) + _dt.timedelta(minutes=s['end_min'])
        conds = ', '.join(dict.fromkeys(b['condition'] for b in s['blocks']))
        summary = f'{doc["name"]}: {s["id"]} {s["operator"]}, {s["episodes"]} episodes ({conds})'
        desc = '\n'.join(f'[ ] {c["text"]}' for c in s['checklist'])
        out.append({'uid': f'{dg}-{s["id"].lower()}@dexvantage.invalid', 'start': t0, 'end': t1, 'summary': summary,
                    'description': desc})
    return out


def to_ics(doc, start_date=None, dtstamp: _dt.datetime | None = None) -> str:
    """The plan as an iCalendar file (RFC 5545): one VEVENT per session, floating local times, CRLF lines folded at
    75 octets; DTSTAMP is dtstamp (UTC; default now)."""
    from .. import __version__
    stamp = (dtstamp or _dt.datetime.now(_dt.timezone.utc)).astimezone(_dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    lines = ['BEGIN:VCALENDAR', 'VERSION:2.0', f'PRODID:-//DexVantage//Campaign planner {__version__}//EN',
             'CALSCALE:GREGORIAN', f'X-WR-CALNAME:{_ics_text(doc["name"])}']
    for e in ics_events(doc, start_date):
        lines += ['BEGIN:VEVENT', f'UID:{e["uid"]}', f'DTSTAMP:{stamp}',
                  f'DTSTART:{e["start"].strftime("%Y%m%dT%H%M%S")}', f'DTEND:{e["end"].strftime("%Y%m%dT%H%M%S")}',
                  f'SUMMARY:{_ics_text(e["summary"])}', f'DESCRIPTION:{_ics_text(e["description"])}',
                  'END:VEVENT']
    lines.append('END:VCALENDAR')
    return ''.join(_fold(x) for x in lines)


_CSS = """
:root{--ink:#1d1d1f;--muted:#5f6368;--line:#d0d4da;--good:#0ca30c;--warning:#c88a00;--critical:#d03b3b}
body{font:13px/1.45 'IBM Plex Sans',system-ui,sans-serif;color:var(--ink);background:#fff;margin:24px;max-width:1100px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:22px 0 6px}h3{font-size:13px;margin:14px 0 4px}
.mono,td.n{font-family:'IBM Plex Mono',ui-monospace,monospace}.muted{color:var(--muted)}
.head{font-size:22px;font-family:'IBM Plex Mono',ui-monospace,monospace}
.st{font-size:12px;padding:1px 6px;border-radius:3px;color:#fff}
.good{background:var(--good)}.warning{background:var(--warning)}.critical{background:var(--critical)}
table{border-collapse:collapse;width:100%;margin:4px 0 8px}th,td{border-bottom:1px solid var(--line);padding:3px 6px;
text-align:left;vertical-align:top}td.n{text-align:right}ul.check{list-style:none;padding-left:0}
ul.check li:before{content:'\\2610  '}
.session{break-inside:avoid;border-top:1px solid var(--line);padding-top:6px}
@media print{body{margin:10mm}.day{break-before:page}.day:first-of-type{break-before:auto}}
"""


def to_html(doc) -> str:
    """A printable page: the headline, the conditions, the design, the totals and budget, then each day's sessions
    with their checklists (one day per printed page)."""
    e = html.escape
    h = doc['headline']
    out = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
           f'content="width=device-width, initial-scale=1"><title>{e(doc["name"])}</title><style>{_CSS}</style>'
           '</head><body>', f'<h1>{e(doc["name"])}</h1>',
           f'<div class="head">{e(h["text"])} <span class="st {e(h["status"])}">{e(h["status_word"])}</span></div>',
           f'<p class="muted">{e(h["explain"])}</p>', '<h2>Conditions</h2><table><tr><th>Id</th>'
           + ''.join(f'<th>{e(AXIS_TITLE[a])}</th>' for a in AXES)
           + '<th>Episode</th><th>Recorded</th><th>Expected kept</th><th>GB each</th></tr>']
    for c in doc['conditions']:
        out.append(f'<tr><td>{e(c["id"])}</td>' + ''.join(f'<td>{e(str(c[a]))}</td>' for a in AXES)
                   + f'<td class="n">{c["episode_s"]:g} s</td><td class="n">{c["recorded"]:,}</td>'
                   f'<td class="n">{c["expected_kept"]:,.1f}</td><td class="n">{c["gb_per_episode"]:.3f}</td></tr>')
    out.append('</table>')
    d = doc['design']
    out.append(f'<h2>Design</h2><p>{e(d["text"])}</p>')
    t = doc['totals']
    out.append('<h2>Totals and budget</h2><table>'
               f'<tr><td>Episodes</td><td class="n">{t["episodes"]:,}</td></tr>'
               f'<tr><td>Days</td><td class="n">{t["days"]}</td></tr>'
               f'<tr><td>Sessions</td><td class="n">{t["sessions"]}</td></tr>'
               f'<tr><td>Hours on the rig</td><td class="n">{t["rig_hours"]:.2f} h</td></tr>'
               f'<tr><td>Storage</td><td class="n">{"≥ " if t["storage_bound"] else ""}{t["storage_gb"]:,.2f} GB</td>'
               '</tr>')
    for c in doc['budget']['checks']:
        out.append(f'<tr><td>{e(c["what"])} (limit {c["limit"]:g} {e(c["unit"])})</td><td class="n">'
                   f'{c["value"]:,.2f} {e(c["unit"])} <span class="st {e(c["status"])}">'
                   f'{e(WORDS[c["status"]])}</span></td></tr>')
    out.append('</table>')
    for a in doc['advice']:
        out.append(f'<p>{e(a)}</p>')
    for day in doc['days']:
        out.append(f'<section class="day"><h2>Day {day["day"]}' + (f' · {e(day["date"])}' if day['date'] else '')
                   + f'</h2><p class="muted">{_plural(day["sessions"], "session")}, '
                   f'{_plural(day["episodes"], "episode")}, {day["gb"]:,.2f} GB</p>')
        for s in doc['sessions']:
            if s['day'] != day['day']:
                continue
            out.append(f'<div class="session"><h3>{e(s["id"])} · {e(s["start"])}–{e(s["end"])} · '
                       f'{e(s["operator"])} · {s["episodes"]} episodes</h3><ul class="check">'
                       + ''.join(f'<li>{e(c["text"])}</li>' for c in s['checklist']) + '</ul></div>')
        out.append('</section>')
    out.append('<h2>Assumptions and defaults</h2><ul>' + ''.join(f'<li>{e(a)}</li>' for a in doc['assumptions'])
               + (f'<li>Left at their DexVantage default: {e(", ".join(doc["defaults"]))}.</li>' if doc['defaults']
                  else '') + '</ul>')
    out.append('<h2>References</h2><ul>' + ''.join(f'<li>{e(r)}</li>' for r in doc['references']) + '</ul>')
    out.append('</body></html>\n')
    return ''.join(out)


def write_files(doc, base: str, start_date=None, dtstamp=None) -> list:
    """<base>.csv, <base>.ics and <base>.html; the paths written."""
    paths = []
    for ext, text, nl in (('.csv', to_csv(doc), None), ('.ics', to_ics(doc, start_date, dtstamp), ''),
                          ('.html', to_html(doc), None)):
        p = base + ext
        try:
            with open(p, 'w', encoding='utf-8', newline=nl) as fh:
                fh.write(text)
        except OSError as e:
            raise CampaignError(f'Could not write {os.path.basename(p)}: {e.strerror or e}.') from None
        paths.append(p)
    return paths


# ====================================================================== into the project
def compact_plan(doc) -> dict:
    """What Apply keeps in the project: the schedule without the checklists (they are re-made from the project)."""
    return {'schema': PLAN_SCHEMA, 'digest': plan_digest(doc), 'headline': doc['headline']['text'],
            'status_word': doc['headline']['status_word'], 'design': {k: v for k, v in doc['design'].items()
                                                                       if k != 'symbol_map'},
            'conditions': [{k: c[k] for k in ('id', 'label', *AXES, 'recorded', 'episode_s')}
                           for c in doc['conditions']],
            'sessions': [{k: s[k] for k in ('id', 'day', 'date', 'start', 'end', 'operator', 'calibration',
                                            'episodes')} | {'blocks': [{k: b[k] for k in ('condition', 'episodes',
                                                                                           'first')}
                                                                       for b in s['blocks']]}
                         for s in doc['sessions']],
            'totals': {k: doc['totals'][k] for k in ('episodes', 'days', 'sessions', 'rig_hours', 'storage_gb')}}


def apply_plan(project, doc, spec: dict | None = None) -> None:
    """Write the plan (compact_plan) into project.analysis.studies['campaign']['plan'], with the spec it was made
    from when one is given; the caller makes it one undo step."""
    cur = dict(project.analysis.studies.get(STUDY_ID) or {})
    if spec is not None:
        cur = {**{k: v for k, v in read_spec(spec).items()}}
    cur['plan'] = compact_plan(doc)
    project.analysis.studies[STUDY_ID] = cur


def applied_plan(project) -> dict | None:
    p = (project.analysis.studies.get(STUDY_ID) or {}).get('plan')
    return p if isinstance(p, dict) and p.get('schema') == PLAN_SCHEMA else None


# ====================================================================== batch
def batch(project, spec_path: str, out: str | None = None) -> tuple:
    """--campaign SPEC.json [--campaign-out OUT.json]: the plan on the project with the spec laid over its stored
    one; writes OUT.json (with provenance) and OUT.csv, OUT.ics, OUT.html beside it (default SPEC.plan.*). Returns
    (doc, [paths])."""
    from . import jobs
    from . import provenance as PV
    spec = read_spec(spec_path)
    snap = project.snapshot()
    stored = {k: v for k, v in (snap.analysis.studies.get(STUDY_ID) or {}).items() if k != 'plan'}
    snap.analysis.studies[STUDY_ID] = {**stored, **spec}
    doc = jobs.run_study(STUDY_ID, snap)
    doc = {k: v for k, v in doc.items() if k != 'cached'}
    if out:
        base = out[:-5] if out.lower().endswith('.json') else out
    else:
        sp = spec_path if not str(spec_path).lstrip().startswith('{') else 'campaign.json'
        base = (sp[:-5] if sp.lower().endswith('.json') else sp) + '.plan'
    jp = base + '.json'
    stamped = PV.stamp(doc, snap, output={'option': 'campaign', 'name': os.path.basename(jp), 'in_folder': False})
    try:
        with open(jp, 'w', encoding='utf-8') as fh:
            json.dump(stamped, fh, indent=1, ensure_ascii=False, allow_nan=False)
            fh.write('\n')
    except OSError as e:
        raise CampaignError(f'Could not write {os.path.basename(jp)}: {e.strerror or e}. Choose a folder that exists '
                            'and that you can write to.') from None
    return doc, [jp] + write_files(doc, base)
