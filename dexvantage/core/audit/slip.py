"""Step change against gradual flex in a series of mount estimates (spec §8.2, campaign checks): ported from Vantage
core/online_calib.py (select_model, refine_step, the Huber weights and the leave-one-out influence check, and the
excursion / knock rule of its event builder), with the vehicle covariates (speed, acceleration, yaw rate, time of day)
replaced by the one a lab campaign has: the time since the campaign's first episode (days when the dataset records
when each episode was taken, else the episode's place in the order). A step is a mount slip, a linear term in time a
gradual drift (a bracket creeping, a clamp relaxing).

    select_model(t, y, sigma, covs=None, ...)   the per-episode estimates of one axis explained by a constant plus
                                                steps and linear covariate terms, chosen greedily by the Bayesian
                                                information criterion, then pruned (Vantage's code, unchanged)
    refine_step(frames_t, frames_y, tau, ...)   a step's time refined from finer estimates around it (unchanged)
    classify(t, y, sigma, ...) -> dict          one axis: {'kind': 'steady' | 'step' | 'excursion' | 'gradual' |
                                                'step and gradual', 'steps', 'flex', 'events', ...}

The model (Vantage's, kept): y_i = level + Σ_s size_s · [i ≥ j_s] + Σ_c slope_c · cov_c(i) + e_i, the per-episode σ_i
giving the relative weights (w_i = 1/σ_i²); a term is added while it lowers BIC = n·ln(RSS/n) + p·ln n by more than
the margin (6: "strong" evidence after Kass and Raftery 1995), a step costing two parameters (size and place) and
needing MIN_SEG estimates on either side, a pair of steps tried at once when no single one pays (a shift that comes
back); then any step within 3 σ of zero or under min_step, and any slope within 3 σ of zero (also with any one
episode left out) or whose half-range effect is under min_amp, is dropped, the weakest first. A step followed by one
back to the earlier level (within 2 σ of the two sizes combined, or min_step) is an excursion (a temporary shift, e.g.
a camera knocked and re-seated); a step that stays is a slip.

What changed from Vantage: the covariate names and units (time 'days' or 'episodes'); MIN_SEG = 2 here (Vantage's 3
is per window of a drive; a campaign has few episodes, and two on either side is the least that tells a step from one
outlying episode); SCALE_FLOOR is in the series' own unit times the stated σ (Vantage floors the residual scale at
0.5 of the stated σ in degrees; here the floor is FLOOR_SHARE of the median σ, so millimetres and degrees both work).

Known limits: one axis at a time (the event builder groups axes that step at the same episode); the time covariate
is linear (a drift that speeds up is fitted as steps); a campaign of fewer than 2·MIN_SEG + 1 episodes cannot show a
step (said in the result); the estimates' σ are taken as given.
"""
from __future__ import annotations

import math

import numpy as np

MIN_SEG = 2                 # estimates on either side of a step (Vantage: 3 windows)
MAX_TERMS = 6               # terms a model may add to its constant (Vantage's)
BIC_MARGIN = 6.0            # Kass and Raftery's "strong" evidence (Vantage's default)
HUBER_K = 1.5               # Vantage's default
FLOOR_SHARE = 0.5           # the residual scale is kept from falling below this share of the median stated σ


def huber_w(e, k):
    """Huber weights of whitened residual norms e (ported from Vantage core/online_calib.py unchanged)."""
    e = np.asarray(e, float)
    return np.where(e <= k, 1.0, k / np.maximum(e, 1e-12))


def _wls(X, y, w):
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    r = y - X @ beta
    return beta, float(np.sum(w * r * r)), r


def select_model(t, y, sigma, covs=None, rho=1.0, margin=BIC_MARGIN, min_seg=MIN_SEG, max_terms=MAX_TERMS,
                 huber_k=HUBER_K, min_step=0.0, min_amp=0.0, scale_floor=0.5):
    """Ported from Vantage core/online_calib.py select_model (the BIC search and its pruning, unchanged but for the
    scale floor, which is passed in: Vantage's module constant SCALE_FLOOR in the unit of y). Returns {'steps':
    [{'j', 'tau', 'size', 'sigma', 'col'}], 'flex': [{'cov', 'slope', 'sigma', 'col'}], 'level', 'level_sigma', 'X',
    'beta', 'cov', 'scale', 'fit', 'n', 'terms', 'bic', 'weights', 'dropped'}."""
    t, y, s = (np.asarray(a, float) for a in (t, y, sigma))
    n = len(y)
    covs = {k: np.asarray(v, float) for k, v in (covs or {}).items()}
    w0 = 1.0 / np.maximum(s, 1e-9) ** 2
    ne = max(rho * n, 1.0)
    floor = float(scale_floor)

    def bic(rss, p):
        return ne * math.log(max(rss / n, floor ** 2)) + p * math.log(max(ne, 1.0 + 1e-9))

    def col_of(term):
        kind, j = term
        if kind == 'level':
            return np.ones(n)
        if kind == 'step':
            return (np.arange(n) >= j).astype(float)
        return covs[j]

    def n_par(terms):
        return sum(2 if k == 'step' else 1 for k, _ in terms)

    def fit(terms, rw):
        X = np.stack([col_of(tm) for tm in terms], 1)
        beta, rss, r = _wls(X, y, w0 * rw)
        return X, beta, rss, r

    def robust(r, rss, p_):
        if n < 3:
            return np.ones(n)
        sc = math.sqrt(max(rss / max(n - p_, 1), floor ** 2))
        return huber_w(np.abs(r) * np.sqrt(w0) / sc, huber_k)

    terms = [('level', None)]
    rw = np.ones(n)
    X, beta, rss, r = fit(terms, rw)
    trace = [bic(rss, 1)]
    while n >= 3 and len(terms) - 1 < max_terms:
        X, _, rss_c, _ = fit(terms, rw)
        base = bic(rss_c, n_par(terms))
        best = None
        taken = [j for k_, j in terms if k_ == 'step']
        for j in range(min_seg, n - min_seg + 1):
            if any(abs(j - jj) < min_seg for jj in taken):
                continue
            _, rss_j, _ = _wls(np.column_stack([X, col_of(('step', j))]), y, w0 * rw)
            b = bic(rss_j, n_par(terms) + 2)
            if best is None or b < best[0]:
                best = (b, ('step', j))
        for name, v in covs.items():
            if ('cov', name) in terms or n < 4 or float(np.std(v)) <= 0:
                continue
            _, rss_c2, _ = _wls(np.column_stack([X, v]), y, w0 * rw)
            b = bic(rss_c2, n_par(terms) + 1)
            if best is None or b < best[0]:
                best = (b, ('cov', name))
        if (best is None or best[0] > base - margin) and n >= 3 * min_seg:
            g = max(1, n // 100)
            js = [j for j in range(min_seg, n - min_seg + 1, g) if all(abs(j - jj) >= min_seg for jj in taken)]
            for a_ in js:
                ca = col_of(('step', a_))
                for b_ in js:
                    if b_ - a_ < min_seg:
                        continue
                    _, rss_p, _ = _wls(np.column_stack([X, ca, col_of(('step', b_))]), y, w0 * rw)
                    b = bic(rss_p, n_par(terms) + 4)
                    if best is None or b < best[0]:
                        best = (b, ('pair', (a_, b_)))
        if best is None or best[0] > base - margin:
            break
        if best[1][0] == 'pair':
            terms += [('step', best[1][1][0]), ('step', best[1][1][1])]
            X, beta, rss, r = fit(terms, rw)
            trace.append(bic(rss, n_par(terms)))
            continue
        terms.append(best[1])
        X, beta, rss, r = fit(terms, rw)
        trace.append(bic(rss, n_par(terms)))

    def final(terms):
        X, beta, rss, r = fit(terms, np.ones(n))
        rw = robust(r, rss, len(terms))
        X, beta, rss, r = fit(terms, rw)
        s2 = max(rss / max(n - len(terms), 1), floor ** 2)
        C = np.linalg.pinv(X.T @ (X * (w0 * rw)[:, None])) * s2 / max(min(rho, 1.0), 1e-6)
        return X, beta, rss, C, rw, s2
    X, beta, rss, C, rw, s2 = final(terms)
    dropped = []
    while len(terms) > 1:
        weak = []
        for i, (kind, j) in enumerate(terms):
            if kind == 'level':
                continue
            se = math.sqrt(max(C[i, i], 0.0))
            z = abs(beta[i]) / max(se, 1e-12)
            if kind == 'step':
                bad = abs(beta[i]) < min_step or z <= 3.0
            else:
                lo, hi = np.percentile(covs[j], [5, 95]) if len(covs[j]) else (0.0, 0.0)
                bad = 0.5 * abs(beta[i]) * (hi - lo) < min_amp or z <= 3.0
                if not bad:
                    z = min(z, _loo_z(X, y, w0 * rw, i, rho, floor))
                    bad = z <= 3.0
            if bad:
                weak.append((z, i))
        if not weak:
            break
        _, i = min(weak)
        dropped.append(terms[i])
        terms = terms[:i] + terms[i + 1:]
        X, beta, rss, C, rw, s2 = final(terms)
    out = {'steps': [], 'flex': [], 'level': float(beta[0]), 'level_sigma': float(math.sqrt(max(C[0, 0], 0.0))),
           'X': X, 'beta': beta, 'cov': C, 'scale': math.sqrt(s2), 'fit': X @ beta, 'n': n, 'terms': terms,
           'bic': trace, 'weights': rw, 'dropped': dropped}
    for i, (kind, j) in enumerate(terms):
        if kind == 'step':
            out['steps'].append({'j': int(j), 'tau': float(0.5 * (t[j - 1] + t[j])), 'size': float(beta[i]),
                                 'sigma': float(math.sqrt(max(C[i, i], 0.0))), 'col': i})
        elif kind == 'cov':
            out['flex'].append({'cov': j, 'slope': float(beta[i]), 'sigma': float(math.sqrt(max(C[i, i], 0.0))),
                                'col': i})
    out['steps'].sort(key=lambda st: st['j'])
    return out


def _loo_z(X, y, w, i, rho, floor):
    """The smallest |β_i| / σ_i over the fits that leave one estimate out (Vantage's influence check)."""
    n, p_ = X.shape
    if n - 1 <= p_:
        return 0.0
    zs = []
    for k in range(n):
        wk = w.copy()
        wk[k] = 0.0
        beta, rss, _ = _wls(X, y, wk)
        s2 = max(rss / max(n - 1 - p_, 1), floor ** 2)
        C = np.linalg.pinv(X.T @ (X * wk[:, None])) * s2 / max(min(rho, 1.0), 1e-6)
        zs.append(abs(beta[i]) / max(math.sqrt(max(C[i, i], 0.0)), 1e-12))
    return float(min(zs))


def refine_step(frames_t, frames_y, tau, before, after, lo, hi):
    """Ported from Vantage core/online_calib.py refine_step, unchanged: the step time refined from finer estimates
    (t, y) between lo and hi, the split with the least robust cost; (tau, resolution)."""
    t = np.asarray(frames_t, float)
    y = np.asarray(frames_y, float)
    m = (t >= lo) & (t <= hi) & np.isfinite(y)
    t, y = t[m], y[m]
    if len(t) < 4:
        return tau, None
    o = np.argsort(t)
    t, y = t[o], y[o]
    mad = 1.4826 * float(np.median(np.abs(y - np.where(np.abs(y - before) < np.abs(y - after), before, after))))
    c = max(mad, 0.25 * abs(after - before), 1e-6)

    def rho(e):
        a = np.abs(e) / c
        return np.where(a <= 1.5, 0.5 * a * a, 1.5 * a - 0.5 * 1.5 ** 2)
    best, cand = None, []
    for i in range(2, len(t) - 1):
        cost = float(np.sum(rho(y[:i] - before)) + np.sum(rho(y[i:] - after)))
        cand.append((cost, i))
        if best is None or cost < best[0]:
            best = (cost, i)
    if best is None:
        return tau, None
    ties = [i for cst, i in cand if cst <= best[0] + 1e-9]
    i = ties[len(ties) // 2]
    res = float(np.median(np.diff(t))) if len(t) > 1 else None
    return float(0.5 * (t[i - 1] + t[i])), res


def events_of(model: dict, t, min_step: float = 0.0, min_amp: float = 0.0, covs: dict | None = None) -> list:
    """The steps and slopes of a select_model result as events, by Vantage's rule (core/online_calib.py _events): a
    step later followed by one of the opposite sign back to within max(2 σ combined, min_step) of the level before it
    is an 'excursion' (it returned), a step that stays a 'step' (Vantage: knock), a slope beyond 3 σ and min_amp a
    'gradual' drift (Vantage: flex). [{'kind', 'j', 'tau', 'size', 'sigma', 'before', 'after', ...}]."""
    t = np.asarray(t, float)
    beta = model['beta']
    steps = [s for s in model['steps'] if abs(s['size']) >= min_step and abs(s['size']) > 3 * s['sigma']]
    levels = []
    for s_ in model['steps']:
        before = beta[0] + sum(o['size'] for o in model['steps'] if o['j'] < s_['j'])
        levels.append((s_, before, before + s_['size']))
    out, used = [], set()
    for i, (s_, before, after) in enumerate(levels):
        if s_ not in steps or id(s_) in used:
            continue
        ev = {'kind': 'step', 'j': s_['j'], 'tau': s_['tau'], 'size': s_['size'], 'sigma': s_['sigma'],
              'before': float(before), 'after': float(after)}
        for s2, _b2, a2 in levels[i + 1:]:
            if s2 not in steps or np.sign(s2['size']) == np.sign(s_['size']):
                break
            if abs(a2 - before) <= max(2 * math.hypot(s_['sigma'], s2['sigma']), min_step):
                used.add(id(s2))
                ev.update({'kind': 'excursion', 'j_end': s2['j'], 'tau_end': s2['tau']})
            break
        out.append(ev)
    for fx in model['flex']:
        v = np.asarray((covs or {}).get(fx['cov'], t), float)
        lo_, hi_ = np.percentile(v, [5, 95]) if len(v) else (0.0, 0.0)
        amp = 0.5 * abs(fx['slope']) * (hi_ - lo_)
        if amp < min_amp or abs(fx['slope']) <= 3 * fx['sigma']:
            continue
        out.append({'kind': 'gradual', 'cov': fx['cov'], 'slope': fx['slope'], 'sigma': fx['sigma'],
                    'amplitude': float(amp), 'total': float(fx['slope'] * (hi_ - lo_)), 'range': [float(lo_),
                                                                                              float(hi_)]})
    return out


def classify(t, y, sigma, *, min_step: float = 0.0, min_amp: float = 0.0, margin: float = BIC_MARGIN,
             min_seg: int = MIN_SEG, cov_name: str = 'time') -> dict:
    """One axis of a campaign: the estimates y (σ sigma) at times t (days or episode order) explained by steps and a
    linear drift in t. {'kind': 'steady' | 'step' | 'excursion' | 'gradual' | 'step and gradual' | 'too short',
    'events', 'level', 'level_sigma', 'scale', 'fit', 'n', 'dropped'}; JSON-able."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    s = np.asarray(sigma, float)
    ok = np.isfinite(t) & np.isfinite(y) & np.isfinite(s) & (s > 0)
    t, y, s = t[ok], y[ok], s[ok]
    n = len(y)
    base = {'n': int(n), 'events': [], 'level': float(np.median(y)) if n else None, 'level_sigma': None,
            'scale': None, 'fit': [float(v) for v in y], 'dropped': [], 'index': [int(i) for i in np.flatnonzero(ok)]}
    if n < 2 * min_seg + 1:
        return dict(base, kind='too short')
    floor = FLOOR_SHARE * float(np.median(s))
    covs = {cov_name: t - t[0]} if float(np.std(t)) > 0 else {}
    m = select_model(t, y, s, covs, 1.0, margin, min_seg, MAX_TERMS, HUBER_K, min_step, min_amp, floor)
    ev = events_of(m, t, min_step, min_amp, covs)
    kinds = {e['kind'] for e in ev}
    if not kinds:
        kind = 'steady'
    elif 'gradual' in kinds and kinds & {'step', 'excursion'}:
        kind = 'step and gradual'
    elif 'gradual' in kinds:
        kind = 'gradual'
    elif 'step' in kinds:
        kind = 'step'
    else:
        kind = 'excursion'
    return dict(base, kind=kind, events=ev, level=m['level'], level_sigma=m['level_sigma'], scale=m['scale'],
                fit=[float(v) for v in m['fit']], dropped=[[k, j if not isinstance(j, tuple) else list(j)]
                                                          for k, j in m['dropped']])
