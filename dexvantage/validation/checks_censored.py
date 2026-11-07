"""Validation of the experiments' rule for figures a study states only as a bound (core/experiments.py: censored).

- experiments.censored_best: the best-run pick, its 'settled' flag and its 'improves' verdict against brute force. On
  random designs (12 runs and a baseline, each figure exact, at least or at most, with 10, 30 or 60 % of them bounds,
  higher or lower is better) the
  check draws, many times, true figures consistent with every statement (an exact figure as stated; "at least v"
  anywhere in [v, v + 100]; "at most v" in [v − 100, v]; the stated figures lie in 0 to 20) and asks of each draw:
  is the picked run's true figure the best? does it beat the rig as it is? The reference written out here: a pick
  marked settled must be best in every draw, an 'improves' of True (False) must hold (fail) in every draw, and a
  verdict of None must be contradicted by some draw both ways (else the rule withheld a decision the bounds settle).
  The error is the number of statements the draws contradict; tolerance 0 (the rule is a decision, not an
  estimate). The draws include the corners of the intervals (every figure at one end, one figure at the other),
  where a contradiction, if any, shows; the rest are uniform within the intervals.

Known limits: the open side of a bound is cut at 100 units for the draws (five times the figures' range); the rule
itself never assumes a limit.
"""
from __future__ import annotations

import numpy as np

from .suite import Outcome, check

KINDS = (None, 'at_least', 'at_most')
SPAN = 100.0                      # the open side of a bound, cut for the draws (figures lie in 0-20)


def _interval(v, b):
    return (v, v) if b is None else (v, v + SPAN) if b == 'at_least' else (v - SPAN, v)


def _draws(rng, iv, n):
    """n draws of true figures within the intervals, the first ones at the intervals' ends (corners)."""
    lo = np.array([a for a, _ in iv])
    hi = np.array([b for _, b in iv])
    out = [lo, hi]
    for k in range(len(iv)):                       # each run alone at its far end, the rest at the other
        a, b = lo.copy(), hi.copy()
        a[k], b[k] = hi[k], lo[k]
        out += [a, b]
    out += list(rng.uniform(lo, hi, (n, len(iv))))
    return np.array(out)


@check('experiments.censored_best', 'experiments', 'Best run with censored figures against brute force',
       'statements of the best-run rule (settled pick, improves True / False / None) contradicted by true figures '
       'drawn within every stated bound', 'brute force over figures consistent with each statement (written out '
       'here), the censored-data rule of never using a bound as if exact (Helsel 2012)', 0.0, 'statements',
       refs=('helsel2012',))
def _censored_best(ctx):
    from ..core import experiments as EX
    rng = np.random.default_rng(20261107)
    trials = 60 if ctx.quick else 300
    wrong, settled, decided, withheld = [], 0, 0, 0
    for t in range(trials):
        better = 'higher' if t % 2 == 0 else 'lower'
        n = 12
        Y = np.round(rng.uniform(0, 20, (n, 1)), 1)
        pb = (0.1, 0.3, 0.6)[t % 3]                 # the share of bounded figures: few to most
        kind = (lambda: None if rng.random() >= pb else KINDS[1 + int(rng.integers(0, 2))])
        B = [[kind()] for _ in range(n)]
        base_v, base_b = float(np.round(rng.uniform(0, 20), 1)), kind()
        bi = EX._best(Y, 0, better, B)
        if bi is None:
            continue
        cont = EX._contenders(Y, 0, better, B, bi)
        imp = EX._improves(float(Y[bi, 0]), B[bi][0], base_v, base_b, better)
        iv = [_interval(float(Y[i, 0]), B[i][0]) for i in range(n)] + [_interval(base_v, base_b)]
        D = _draws(rng, iv, 200)
        s = 1.0 if better == 'higher' else -1.0
        runs, base = s * D[:, :n], s * D[:, n]
        is_best = runs[:, bi] >= runs.max(axis=1) - 1e-12
        beats = runs[:, bi] > base + 1e-12
        if not cont:
            settled += 1
            if not is_best.all():
                wrong.append({'trial': t, 'statement': 'settled', 'run': bi + 1})
        if imp is True and not beats.all():
            wrong.append({'trial': t, 'statement': 'improves', 'run': bi + 1})
        if imp is False and beats.any():
            wrong.append({'trial': t, 'statement': 'does not improve', 'run': bi + 1})
        if imp is None:
            withheld += 1
            if beats.all() or not beats.any():
                wrong.append({'trial': t, 'statement': 'not settled although the bounds decide it', 'run': bi + 1})
        else:
            decided += 1
    return Outcome(float(len(wrong)), {'trials': trials, 'settled_picks': settled, 'verdicts_decided': decided,
                                       'verdicts_withheld': withheld, 'contradicted': wrong[:12]},
                   note=f'{trials} random designs of 12 runs; draws include every interval\'s ends')
