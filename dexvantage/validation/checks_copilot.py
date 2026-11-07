"""Checks of the Assistant dock's rules (assistant/numcheck.py, copilot.py, edits.py), each against an independent
reference written out here.

- copilot.number_check: the number post-check on a hand-labelled corpus: answers with the figures a reader would flag
  written out beside them, from figures copied from tool results (as written, rounded, as a percentage, in mm, in
  degrees, without the sign), invented ones, and numbers that are names (UR5e, SO-100, v2.1, a date, a step number).
  The error is the number of answers whose flagged set differs from the labels; the tolerance is 0 because the rule
  is deterministic: any disagreement is a bug.
- copilot.plan_gate: the approval gate (copilot.gate, verify) on records intact and altered (settings changed after
  approval, a step's kind changed, the stored hash replaced, never approved, no approval time, another version, a step
  that does not exist, other settings at run time, no record, no ticket). The reference is the decision each case must
  get and the records' SHA-256 recomputed here with hashlib over canonical JSON (sorted keys, no spaces; FIPS 180-4).
  The error counts wrong decisions and hash mismatches; tolerance 0.
- copilot.proposal_roundtrip: an edit proposal applied section by section (edits.apply, the undo stack's unit) and
  taken back by restoring the sections as they were, on the SO-100 sample: after applying, the project must equal
  core/factors.apply's; after restoring, the original, field for field (canonical JSON of Project.to_dict). The error
  counts differing sections; tolerance 0.
"""
from __future__ import annotations

import copy
import hashlib
import json

from .suite import Outcome, check

# (answer, sources, the figures a reader flags): labelled by hand
CORPUS = (
    ('Coverage is 94.5 % seen once or more.', ['{"seen_ge1": 0.945208}'], set()),
    ('Coverage is 94.52 %.', ['{"seen_ge1": 0.945208}'], set()),
    ('Coverage is 95 %.', ['{"seen_ge1": 0.945208}'], set()),
    ('Coverage is 96 %.', ['{"seen_ge1": 0.945208}'], {'96'}),
    ('The wrist camera sits 60 mm below the jaw.', ['{"position_m": [-0.01, 0.0, -0.06]}'], set()),
    ('It is pitched −23.2°.', ['{"rpy_deg": [0.0, -23.199, -90.0]}'], set()),
    ('It is pitched 23.2° down.', ['{"rpy_deg": [0.0, -23.199, -90.0]}'], set()),
    ('It is pitched 25°.', ['{"rpy_deg": [0.0, -23.199, -90.0]}'], {'25'}),
    ('The joint turns 0.5236 rad, that is 30°.', ['{"q": 0.5236}'], set()),
    ('The D405\'s minimum depth is 70 mm.', ['{"min_z_m": null, "note": "not stated by the maker"}'], {'70'}),
    ('The maker does not state the minimum depth.', ['{"min_z_m": null}'], set()),
    ('The skew is 54.4 ms on the IR left stream.', ['{"worst_skew_s": 0.0544258}'], set()),
    ('The skew is 45 ms.', ['{"worst_skew_s": 0.0544258}'], {'45'}),
    ('1 886 tests passed.', ['{"passed": 1886}'], set()),
    ('2 000 frames were read.', ['{"frames": 1886}'], {'2 000'}),
    ('The UR5e and the SO-100 run on claude-opus-5-5 (v2.1, §7.3).', [], set()),
    ('Read on 2026-11-07 at 10:42.', [], set()),
    ('Step 3 of 4 uses 2 cameras.', [], set()),
    ('The blind gap is 5 mm.', ['{"gap_m": null}'], {'5'}),            # a small integer with a unit is a figure
    ('The gap is 5 mm.', ['{"gap_m": 0.005}'], set()),
    ('It runs 8 ms behind.', ['{"skew_s": 0.0544}'], {'8'}),
    ('A value of 3.2e-5 m.', ['{"err": 3.2e-05}'], set()),
    ('A value of 3.3e-5 m.', ['{"err": 3.2e-05}'], {'3.3e-5'}),
    ('Inference takes 120 ms per chunk.', ['{"inference_s": null}'], {'120'}),
    ('See `0.25` in the settings and https://example.org/a/12.5', [], set()),
    ('1. First item\n2. Second item', [], set()),
    ('The budget is 3 600 Mbit/s.', ['{"capacity_mbps": 3600}'], set()),
    ('The sensor uid 3e512abc reads 0.42.', ['{"uid": "3e512abc", "v": 0.42}'], set()),
)


@check('copilot.number_check', 'copilot', 'Number post-check against a hand-labelled corpus',
       'answers whose flagged figures differ from the labels', 'a corpus of answers labelled by hand in this check',
       0, 'answers')
def _number_check(ctx):
    from ..assistant import numcheck as NC
    bad = []
    for text, src, want in CORPUS:
        got = {u['text'] for u in NC.unsupported(text, src)['unsupported']}
        if got != want:
            bad.append({'answer': text, 'flagged': sorted(got), 'labelled': sorted(want)})
    return Outcome(float(len(bad)), {'answers': len(CORPUS), 'labelled_flags': sum(len(w) for *_, w in CORPUS),
                                     'mismatches': bad})


def _sha(o) -> str:
    """SHA-256 of canonical JSON, written out again here (sorted keys, no spaces, UTF-8)."""
    return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                                     default=str).encode('utf-8')).hexdigest()


@check('copilot.plan_gate', 'copilot', 'Copilot approval gate on intact and altered records',
       'wrong gate decisions plus SHA-256 mismatches', 'the decision each case must get; SHA-256 by hashlib', 0,
       'cases', refs=('fips180_4',))
def _plan_gate(ctx):
    from ..assistant import copilot as CP
    p = ctx.project
    plan = CP.build(p, [{'kind': 'study', 'study': 'bandwidth', 'settings': {'margin': 0.2}},
                        {'kind': 'study', 'study': 'coverage'}], 'Validation plan')
    rec, errors = CP.approve(p, plan, 'validation', 'none')
    if rec is None:
        return Outcome(float('inf'), {'errors': errors}, ok=False)
    hash_bad = sum(s['settings_sha256'] != _sha({'kind': s['kind'], 'settings': s['settings']})
                   for s in rec['steps'])
    hash_bad += rec['settings_sha256'] != _sha([{'kind': s['kind'], 'settings': s['settings']} for s in rec['steps']])
    tk = CP.ticket(rec, 0)
    s0 = rec['steps'][0]['settings']

    def altered(fn):
        r = copy.deepcopy(rec)
        fn(r)
        return r
    cases = [
        ('intact', [rec], tk, s0, True),
        ('the second step', [rec], CP.ticket(rec, 1), rec['steps'][1]['settings'], True),
        ('settings changed after approval', [altered(lambda r: r['steps'][0]['settings']['settings'].update(
            margin=0.9))], tk, dict(s0, settings={'margin': 0.9}), False),
        ('kind changed', [altered(lambda r: r['steps'][0].update(kind='export'))], tk, s0, False),
        ('hash replaced', [altered(lambda r: r.update(settings_sha256='0' * 64))], tk, s0, False),
        ('never approved', [altered(lambda r: r.update(status='draft'))], tk, s0, False),
        ('no approval time', [altered(lambda r: r.update(approved_utc=''))], tk, s0, False),
        ('another version', [rec], dict(tk, version=2), s0, False),
        ('no such step', [rec], dict(tk, step=5), s0, False),
        ('other settings at run time', [rec], tk, dict(s0, settings={'margin': 0.21}), False),
        ('no record', [], tk, s0, False),
        ('no ticket', [rec], None, s0, False),
    ]
    wrong = []
    for name, rs, t, settings, want in cases:
        got = CP.gate(rs, t, 'study', settings)[0]
        if got != want:
            wrong.append({'case': name, 'gate': got, 'expected': want})
    return Outcome(float(len(wrong) + hash_bad), {'cases': len(cases), 'wrong_decisions': wrong,
                                                  'hash_mismatches': int(hash_bad)})


def _canon(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(',', ':'), default=str)


@check('copilot.proposal_roundtrip', 'copilot', 'Edit proposal applied and taken back',
       'sections differing from core.factors.apply after applying, plus top-level fields differing from the original '
       'after restoring', 'core/factors.apply and the project\'s own serialisation (Project.to_dict)', 0, 'sections')
def _proposal_roundtrip(ctx):
    from ..assistant import edits as ED
    from ..core import factors as F
    from ..core.robot.samples import sample_project
    base = sample_project('so100')
    s = base.variant.sensors[0]
    values = {f'sensor:{s.uid}.z': 0.05, f'sensor:{s.uid}.dpitch': -5.0, f'sensor:{s.uid}.enabled': False}
    prop = ED.from_factors(base, 'validation', values)
    want = F.apply(base, values)
    live = base.snapshot()
    keys = ED.keys_for(live, prop)
    before = {k: copy.deepcopy(live.section(k)) for k in keys}
    ED.apply(live, prop)
    applied_bad = sum(_canon(live.section(k)) != _canon(want.section(k)) for k in ED.SECTIONS)
    for k in keys:
        live.restore(k, copy.deepcopy(before[k]))
    a, b = live.to_dict(), base.to_dict()
    a.pop('log', None)
    b.pop('log', None)
    restored_bad = sum(_canon(a.get(k)) != _canon(b.get(k)) for k in set(a) | set(b))
    return Outcome(float(applied_bad + restored_bad), {'factors': list(values), 'sections': keys,
                                                       'after_apply_differs': int(applied_bad),
                                                       'after_restore_differs': int(restored_bad)})
