"""NSGA-II in NumPy (Deb, Pratap, Agarwal and Meyarivan 2002): an evolutionary multi-objective optimiser over mixed
continuous and categorical variables, with constraints, a budget in evaluations, a seed, progress and cancel.

    Var(name, lo, hi)                       a continuous variable in [lo, hi]
    Var(name, choices=[...])                a categorical variable (its gene is the index of the choice)
    Problem(vars, n_obj, evaluate, n_con=0, batch=False)
        evaluate(x) -> (f, g)               one decoded genome (a list of values) -> objectives f (n_obj,), all
                                            MINIMISED, and constraint violations g (n_con,), each >= 0 (0 = met);
                                            None stops the run (cancelled part way: nothing recorded)
        batch=True: evaluate(X) -> (F, G)   the whole (n, n_var) gene array at once (the ZDT benchmarks)
    run(problem, pop=..., budget=..., seed=..., progress=fn(frac, text), cancel=fn() -> bool, initial=[x, ...])
        -> Result: the final population (X, F, CV, rank), the archive of every evaluated genome, the evaluations
           used and whether it was cancelled.

The algorithm, as in the paper and Deb's reference C implementation (Kanpur Genetic Algorithms Laboratory):

1. A random initial population of `pop` genomes (the given initial genomes first), evaluated.
2. Each generation: binary tournament selection on the crowded comparison (lower rank wins, then the larger crowding
   distance); for continuous genes simulated binary crossover (SBX, Deb and Agrawal 1995) with probability pc per pair
   and 0.5 per gene, distribution index eta_c, in its bounded form; then polynomial mutation (Deb and Goyal 1996),
   bounded form, probability pm per gene (default 1 / number of genes), distribution index eta_m. Categorical genes:
   uniform crossover (each gene swapped with probability 0.5 when the pair crosses) and mutation to another choice
   drawn uniformly, with the same pm.
3. The offspring are evaluated; parents and offspring are merged and sorted into non-domination fronts under
   CONSTRAINED domination (Deb 2002, section VI): a feasible genome dominates an infeasible one; of two infeasible
   ones the smaller total violation dominates; of two feasible ones Pareto dominance on the objectives. The next
   population is filled front by front, the last front cut by crowding distance (the larger kept).
4. Stops when the budget of evaluations is spent (the last generation is cut to fit it) or cancel() turns true.

Everything random comes from one numpy Generator seeded with `seed`, so a run is reproducible exactly.

Known limits: NSGA-II's crowding distance spreads well for two or three objectives and less well beyond (many-objective
problems want reference directions, NSGA-III); the result approximates the Pareto front and is not proven optimal; the
budget is in evaluations, not seconds; a NaN objective is treated as +inf (worst) by the caller's choice, here only
in sorting.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class Var:
    name: str
    lo: float = 0.0
    hi: float = 1.0
    choices: list | None = None          # categorical when given

    @property
    def categorical(self) -> bool:
        return self.choices is not None


@dataclass
class Problem:
    vars: list
    n_obj: int
    evaluate: Callable
    n_con: int = 0
    batch: bool = False

    def decode(self, x) -> list:
        """A gene row -> the variables' values (a categorical gene as its choice)."""
        out = []
        for v, g in zip(self.vars, x):
            out.append(v.choices[int(round(g))] if v.categorical else float(g))
        return out


@dataclass
class Result:
    X: np.ndarray                    # (pop, n_var) the final population's genes
    F: np.ndarray                    # (pop, n_obj) minimised objectives
    CV: np.ndarray                   # (pop,) total constraint violation (0 = feasible)
    rank: np.ndarray                 # (pop,) front index (0 = the non-dominated front)
    archive_X: np.ndarray            # every evaluated genome, in evaluation order
    archive_F: np.ndarray
    archive_CV: np.ndarray
    evaluations: int
    generations: int
    cancelled: bool = False
    history: list = field(default_factory=list)       # per generation: {'evaluations', 'front', 'feasible'}

    def front(self, archive: bool = False) -> np.ndarray:
        """Indices of the first front (into the final population, or into the archive)."""
        if archive:
            F, CV = self.archive_F, self.archive_CV
        else:
            F, CV = self.F, self.CV
        if not len(F):
            return np.zeros(0, int)
        return non_dominated_sort(F, CV)[0]


# ====================================================================== sorting
def dominance_matrix(F, CV=None) -> np.ndarray:
    """D[i, j] = True when i constrained-dominates j (Deb 2002): feasible beats infeasible; of two infeasible the
    smaller violation; of two feasible Pareto dominance (no worse in every objective, better in one). NaN objectives
    count as +inf."""
    F = np.asarray(F, float)
    F = np.where(np.isnan(F), np.inf, F)
    n = len(F)
    cv = np.zeros(n) if CV is None else np.asarray(CV, float)
    le = (F[:, None, :] <= F[None, :, :]).all(axis=2)
    lt = (F[:, None, :] < F[None, :, :]).any(axis=2)
    pareto = le & lt
    feas = cv <= 0.0
    fi, fj = feas[:, None], feas[None, :]
    return (fi & ~fj) | (~fi & ~fj & (cv[:, None] < cv[None, :])) | (fi & fj & pareto)


def non_dominated_sort(F, CV=None) -> list:
    """The fronts [indices of front 0, front 1, ...] under constrained domination (the fast sort's result: every
    member of front k is dominated only by members of fronts < k)."""
    D = dominance_matrix(F, CV)
    n = len(D)
    count = D.sum(axis=0)                 # how many dominate each
    left = np.ones(n, bool)
    fronts = []
    while left.any():
        cur = np.flatnonzero(left & (count == 0))
        if not len(cur):                  # cannot happen for a strict partial order; guard against NaN surprises
            cur = np.flatnonzero(left)
        fronts.append(cur)
        left[cur] = False
        count = count - D[cur].sum(axis=0)
    return fronts


def crowding_distance(F) -> np.ndarray:
    """Per member of one front: the sum over objectives of the normalised gap between its two neighbours along that
    objective; the extremes of each objective get +inf (Deb 2002, section III-B). A member whose objective is not a
    finite number (a figure not stated: +inf, or NaN) takes no part along that objective: the extremes are those of
    the stated values, and it gains nothing there (giving it +inf would keep unstated designs ahead of stated ones in
    the crowded comparison, the opposite of ranking them last)."""
    F = np.asarray(F, float)
    n = len(F)
    F = F.reshape(n, -1)
    m = F.shape[1]
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for k in range(m):
        col = F[:, k]
        ok = np.isfinite(col)
        if not ok.any():
            continue
        order = np.argsort(np.where(ok, col, np.inf), kind='stable')
        order = order[:ok.sum()]
        lo, hi = col[order[0]], col[order[-1]]
        d[order[0]] = d[order[-1]] = np.inf
        span = hi - lo
        if span <= 0:
            continue
        d[order[1:-1]] += (col[order[2:]] - col[order[:-2]]) / span
    return d


def rank_and_crowd(F, CV):
    fronts = non_dominated_sort(F, CV)
    rank = np.zeros(len(F), int)
    crowd = np.zeros(len(F))
    for k, fr in enumerate(fronts):
        rank[fr] = k
        crowd[fr] = crowding_distance(np.asarray(F)[fr])
    return rank, crowd, fronts


def select(F, CV, n: int) -> np.ndarray:
    """Indices of the n survivors: whole fronts in order, the last one cut by crowding distance (largest kept)."""
    fronts = non_dominated_sort(F, CV)
    keep = []
    for fr in fronts:
        if len(keep) + len(fr) <= n:
            keep += list(fr)
            continue
        cd = crowding_distance(np.asarray(F)[fr])
        order = np.argsort(-cd, kind='stable')
        keep += list(fr[order[:n - len(keep)]])
        break
    return np.array(keep, int)


# ====================================================================== variation
def tournament(rank, crowd, n: int, rng) -> np.ndarray:
    """n parents by binary tournament on the crowded comparison."""
    N = len(rank)
    a = rng.integers(0, N, n)
    b = rng.integers(0, N, n)
    a_wins = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] > crowd[b]))
    tie = (rank[a] == rank[b]) & (crowd[a] == crowd[b])
    coin = rng.random(n) < 0.5
    return np.where(a_wins | (tie & coin), a, b)


def sbx(p1, p2, lo, hi, eta: float, pc: float, rng):
    """Simulated binary crossover, bounded (Deb's nsga2 C code, realcross): two parent arrays (n, d) -> two children.
    Each pair crosses with probability pc; each gene of a crossing pair with probability 0.5 when the parents differ."""
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    n, d = p1.shape
    c1, c2 = p1.copy(), p2.copy()
    if d == 0:
        return c1, c2
    lo = np.broadcast_to(np.asarray(lo, float), (n, d))
    hi = np.broadcast_to(np.asarray(hi, float), (n, d))
    pair = rng.random(n) < pc
    gene = rng.random((n, d)) < 0.5
    u = rng.random((n, d))
    swap = rng.random((n, d)) < 0.5
    y1 = np.minimum(p1, p2)
    y2 = np.maximum(p1, p2)
    diff = y2 - y1
    do = pair[:, None] & gene & (np.abs(p1 - p2) > 1e-14) & (hi > lo)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        safe = np.where(diff > 0, diff, 1.0)

        def betaq(beta):
            alpha = 2.0 - np.power(beta, -(eta + 1.0))
            return np.where(u <= 1.0 / alpha, np.power(u * alpha, 1.0 / (eta + 1.0)),
                            np.power(1.0 / np.maximum(2.0 - u * alpha, 1e-300), 1.0 / (eta + 1.0)))
        bq1 = betaq(1.0 + 2.0 * (y1 - lo) / safe)
        bq2 = betaq(1.0 + 2.0 * (hi - y2) / safe)
        k1 = 0.5 * ((y1 + y2) - bq1 * diff)
        k2 = 0.5 * ((y1 + y2) + bq2 * diff)
    k1 = np.clip(np.nan_to_num(k1, nan=0.0), lo, hi)
    k2 = np.clip(np.nan_to_num(k2, nan=0.0), lo, hi)
    a = np.where(swap, k2, k1)
    b = np.where(swap, k1, k2)
    c1 = np.where(do, a, c1)
    c2 = np.where(do, b, c2)
    return c1, c2


def polynomial_mutation(x, lo, hi, eta: float, pm: float, rng):
    """Polynomial mutation, bounded (Deb's nsga2 C code, real_mutate_ind): each gene with probability pm."""
    x = np.asarray(x, float).copy()
    n, d = x.shape
    if d == 0:
        return x
    lo = np.broadcast_to(np.asarray(lo, float), (n, d))
    hi = np.broadcast_to(np.asarray(hi, float), (n, d))
    span = hi - lo
    do = (rng.random((n, d)) < pm) & (span > 0)
    u = rng.random((n, d))
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.where(span > 0, span, 1.0)
        d1 = (x - lo) / s
        d2 = (hi - x) / s
        mp = 1.0 / (eta + 1.0)
        v1 = 2.0 * u + (1.0 - 2.0 * u) * np.power(1.0 - d1, eta + 1.0)
        v2 = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * np.power(1.0 - d2, eta + 1.0)
        dq = np.where(u <= 0.5, np.power(np.maximum(v1, 0.0), mp) - 1.0, 1.0 - np.power(np.maximum(v2, 0.0), mp))
    y = np.clip(x + dq * span, lo, hi)
    return np.where(do, y, x)


def uniform_cross(p1, p2, pc: float, rng):
    """Uniform crossover of categorical genes: each gene swapped with probability 0.5 when the pair crosses."""
    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    n, d = p1.shape
    if d == 0:
        return p1.copy(), p2.copy()
    sw = (rng.random(n) < pc)[:, None] & (rng.random((n, d)) < 0.5)
    return np.where(sw, p2, p1), np.where(sw, p1, p2)


def categorical_mutation(x, sizes, pm: float, rng):
    """Each gene with probability pm moves to another of its choices, drawn uniformly."""
    x = np.asarray(x, float).copy()
    n, d = x.shape
    if d == 0:
        return x
    sizes = np.asarray(sizes, int)
    do = (rng.random((n, d)) < pm) & (sizes[None, :] > 1)
    step = 1 + np.floor(rng.random((n, d)) * np.maximum(sizes[None, :] - 1, 1))      # 1 … size-1
    y = np.mod(x + step, np.maximum(sizes[None, :], 1))
    return np.where(do, y, x)


# ====================================================================== the loop
def random_genes(problem: Problem, n: int, rng) -> np.ndarray:
    X = np.zeros((n, len(problem.vars)))
    for j, v in enumerate(problem.vars):
        if v.categorical:
            X[:, j] = rng.integers(0, max(1, len(v.choices)), n)
        else:
            X[:, j] = v.lo + rng.random(n) * (v.hi - v.lo)
    return X


def encode(problem: Problem, values) -> np.ndarray:
    """Variable values -> a gene row (a categorical value -> its index; a value not among the choices -> 0)."""
    row = np.zeros(len(problem.vars))
    for j, (v, x) in enumerate(zip(problem.vars, values)):
        if v.categorical:
            row[j] = next((k for k, c in enumerate(v.choices) if c == x), 0)
        else:
            row[j] = min(max(float(x), v.lo), v.hi)
    return row


def run(problem: Problem, pop: int = 40, budget: int = 2000, seed: int = 0, eta_c: float = 15.0,
        eta_m: float = 20.0, pc: float = 0.9, pm: float | None = None, progress=None, cancel=None,
        initial=None, prefetch=None) -> Result:
    """NSGA-II on problem (see the module docstring). budget: evaluations in all (at least pop). prefetch(rows):
    called with a generation's decoded genomes before they are evaluated one by one (core/cluster scores them in
    parallel there and the evaluator then finds them in its memo); it changes no result."""
    rng = np.random.default_rng(int(seed))
    nv = len(problem.vars)
    pop = max(2, int(pop))
    budget = max(pop, int(budget))
    cat = np.array([v.categorical for v in problem.vars], bool)
    lo = np.array([0.0 if v.categorical else v.lo for v in problem.vars], float)
    hi = np.array([float(len(v.choices) - 1) if v.categorical else v.hi for v in problem.vars], float)
    sizes = np.array([len(v.choices) if v.categorical else 0 for v in problem.vars], int)
    pm = (1.0 / max(1, nv)) if pm is None else float(pm)
    aX, aF, aCV = [], [], []
    state = {'evals': 0, 'cancelled': False}

    def stop() -> bool:
        if cancel is not None and cancel():
            state['cancelled'] = True
        return state['cancelled']

    def evaluate(X):
        """(F, CV) of the rows of X; on cancel returns those evaluated so far (fewer rows)."""
        if problem.batch:
            if stop():
                return X[:0], np.zeros((0, problem.n_obj)), np.zeros(0)
            F, G = problem.evaluate(X)
            F = np.asarray(F, float).reshape(len(X), problem.n_obj)
            G = np.asarray(G, float).reshape(len(X), -1) if problem.n_con else np.zeros((len(X), 0))
            done = X
        else:
            rows_f, rows_g, done = [], [], []
            if prefetch is not None and len(X) and not stop():
                prefetch([problem.decode(x) for x in X])
            for x in X:
                if stop():
                    break
                res = problem.evaluate(problem.decode(x))
                if res is None:                  # the evaluator was cancelled part way: nothing recorded
                    state['cancelled'] = True
                    break
                f, g = res
                rows_f.append(np.asarray(f, float).reshape(problem.n_obj))
                rows_g.append(np.asarray(g if g is not None else [], float).reshape(-1))
                done.append(x)
                state['evals'] += 1
                if progress is not None:
                    progress(min(1.0, state['evals'] / budget), f'{state["evals"]} of {budget} evaluations')
            done = np.array(done, float).reshape(-1, nv)
            F = np.array(rows_f, float).reshape(-1, problem.n_obj)
            G = np.array(rows_g, float).reshape(len(F), -1) if rows_g and rows_g[0].size else np.zeros((len(F), 0))
            state['evals'] -= len(done)          # counted once below, as for a batch
        CV = np.clip(G, 0.0, None).sum(axis=1) if G.size else np.zeros(len(F))
        CV = np.where(np.isfinite(CV), CV, 1e12)
        state['evals'] += len(done)
        aX.append(done)
        aF.append(F)
        aCV.append(CV)
        return done, F, CV

    X0 = random_genes(problem, pop, rng)
    if initial is not None:
        init = [encode(problem, x) for x in initial][:pop]
        for k, row in enumerate(init):
            X0[k] = row
    X, F, CV = evaluate(X0)
    gen = 0
    history = []
    while state['evals'] < budget and not state['cancelled'] and len(X) >= 2:
        gen += 1
        rank, crowd, fronts = rank_and_crowd(F, CV)
        n_off = min(pop, budget - state['evals'])
        n_pair = (n_off + 1) // 2
        a = X[tournament(rank, crowd, n_pair, rng)]
        b = X[tournament(rank, crowd, n_pair, rng)]
        c1, c2 = np.empty_like(a), np.empty_like(b)
        k1, k2 = sbx(a[:, ~cat], b[:, ~cat], lo[~cat], hi[~cat], eta_c, pc, rng)
        c1[:, ~cat], c2[:, ~cat] = k1, k2
        k1, k2 = uniform_cross(a[:, cat], b[:, cat], pc, rng)
        c1[:, cat], c2[:, cat] = k1, k2
        kids = np.concatenate([c1, c2])[:n_off]
        kids[:, ~cat] = polynomial_mutation(kids[:, ~cat], lo[~cat], hi[~cat], eta_m, pm, rng)
        kids[:, cat] = categorical_mutation(kids[:, cat], sizes[cat], pm, rng)
        kX, kF, kCV = evaluate(kids)
        XX = np.concatenate([X, kX])
        FF = np.concatenate([F, kF])
        CC = np.concatenate([CV, kCV])
        keep = select(FF, CC, pop)
        X, F, CV = XX[keep], FF[keep], CC[keep]
        front0 = non_dominated_sort(F, CV)[0]
        history.append({'evaluations': state['evals'], 'front': int(len(front0)),
                        'feasible': int((CV <= 0).sum())})
    rank, _crowd, _fr = rank_and_crowd(F, CV) if len(F) else (np.zeros(0, int), None, None)
    cat_ = (lambda parts, w: np.concatenate(parts) if parts else np.zeros((0, w)))
    return Result(X, F, CV, rank, cat_(aX, nv), cat_(aF, problem.n_obj),
                  np.concatenate(aCV) if aCV else np.zeros(0), state['evals'], gen, state['cancelled'], history)


# ====================================================================== benchmarks and measures (validation, tests)
def zdt1(X):
    """ZDT1 (Zitzler, Deb and Thiele 2000): f1 = x1, g = 1 + 9 mean(x2..n), f2 = g (1 - sqrt(f1 / g))."""
    X = np.asarray(X, float)
    f1 = X[:, 0]
    g = 1.0 + 9.0 * X[:, 1:].mean(axis=1)
    return np.stack([f1, g * (1.0 - np.sqrt(f1 / g))], axis=1), np.zeros((len(X), 0))


def zdt2(X):
    """ZDT2: f2 = g (1 - (f1 / g)^2)."""
    X = np.asarray(X, float)
    f1 = X[:, 0]
    g = 1.0 + 9.0 * X[:, 1:].mean(axis=1)
    return np.stack([f1, g * (1.0 - (f1 / g) ** 2)], axis=1), np.zeros((len(X), 0))


def constr(X):
    """Deb's CONSTR problem (Deb 2001, Multi-Objective Optimization Using Evolutionary Algorithms, the constrained test
    problems): minimise f1 = x1, f2 = (1 + x2) / x1, x1 in [0.1, 1], x2 in [0, 5], subject to x2 + 9 x1 >= 6 and
    -x2 + 9 x1 >= 1. The violations returned are max(0, 6 - x2 - 9 x1) and max(0, 1 + x2 - 9 x1)."""
    X = np.asarray(X, float)
    x1, x2 = X[:, 0], X[:, 1]
    F = np.stack([x1, (1.0 + x2) / x1], axis=1)
    G = np.stack([np.maximum(0.0, 6.0 - x2 - 9.0 * x1), np.maximum(0.0, 1.0 + x2 - 9.0 * x1)], axis=1)
    return F, G


def zdt_front(kind: str, n: int = 1000) -> np.ndarray:
    """The analytic Pareto front (g = 1): ZDT1 f2 = 1 - sqrt(f1), ZDT2 f2 = 1 - f1^2, f1 uniform on [0, 1]."""
    f1 = np.linspace(0.0, 1.0, n)
    f2 = 1.0 - np.sqrt(f1) if kind == 'zdt1' else 1.0 - f1 ** 2
    return np.stack([f1, f2], axis=1)


def constr_front(n: int = 1000) -> np.ndarray:
    """CONSTR's analytic front, worked out from the constraints: for x1 >= 2/3 the first constraint is inactive and
    x2 = 0 is best, f2 = 1 / f1; for 7/18 <= x1 < 2/3 the first is active, x2 = 6 - 9 x1, f2 = (7 - 9 f1) / f1 (the
    second constraint needs x1 >= 7/18 there). Points spread evenly along f1 on [7/18, 1]."""
    f1 = np.linspace(7.0 / 18.0, 1.0, n)
    f2 = np.where(f1 >= 2.0 / 3.0, 1.0 / f1, (7.0 - 9.0 * f1) / f1)
    return np.stack([f1, f2], axis=1)


def igd(front, reference) -> float:
    """Inverted generational distance: the mean, over the reference points, of the Euclidean distance to the nearest
    point of the front found (Coello and Reyes Sierra 2004). 0 when the front covers the reference exactly."""
    A = np.asarray(front, float).reshape(-1, np.asarray(reference).shape[1])
    R = np.asarray(reference, float)
    if not len(A):
        return math.inf
    d = np.sqrt(((R[:, None, :] - A[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    return float(d.mean())


def benchmark_problem(kind: str, n_var: int = 30) -> Problem:
    """ZDT1 / ZDT2 (n_var genes on [0, 1]) or CONSTR as batch Problems."""
    if kind == 'constr':
        return Problem([Var('x1', 0.1, 1.0), Var('x2', 0.0, 5.0)], 2, constr, n_con=2, batch=True)
    fn = zdt1 if kind == 'zdt1' else zdt2
    return Problem([Var(f'x{i + 1}', 0.0, 1.0) for i in range(n_var)], 2, fn, batch=True)
