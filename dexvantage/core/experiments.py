"""Designed experiments (spec §7.11): many versions of the rig evaluated automatically, and what drives each figure.

Ported from Vantage core/experiments.py (the designs, the Sobol estimators with their bootstrap, the main effects and
the result layout); the automotive factors (cameras on a vehicle body, load cases, weather, the ROI) are replaced by
DexVantage's factor paths (core/factors.py) and the responses by the named metrics of core/metrics.py.

Factors
-------
A factor is one settable quantity of the project, named by its core/factors path (a sensor's mount x, y, z, roll,
pitch, yaw, a shift dx, dy, dz, a small turn droll, dpitch, dyaw, its library model, whether it is enabled, a study
setting setting:<study>.<key>, or any json:/pointer). It is one of three kinds:

* continuous: a range [lo, hi] (and, for the full factorial, a number of evenly spaced levels; 2 by default); an
  integer flag rounds the value (a count setting);
* categorical: a list of levels (library ids, frame names, any JSON values), taken in the order given;
* bool: the levels false and true.

Designs
-------
* Full factorial: every combination of the factors' levels.
* Fractional factorial, two levels, resolution III or IV (Box, Hunter and Hunter 2005, ch. 6–7): the smallest
  2^(k−p) design with generators from the base factors' interaction columns. Resolution III: n = the smallest power of
  two above k, the added factors set to the two-, then three-factor … interactions of the base factors (main effects
  free of each other, aliased with two-factor interactions). Resolution IV: n = the smallest power of two with
  n ≥ 2k, every added factor set to an interaction of an odd number (≥ 3) of base factors, so every word of the
  defining relation has even length ≥ 4 (main effects free of two-factor interactions). The resolution actually
  reached is found again by brute force over the design's columns (resolution_of) and reported.
* Latin hypercube (McKay, Beckman and Conover 1979): n points, each factor's range cut into n equal strata with one
  point in each, the strata paired at random (seeded).
* Sobol sequence: the first n points of a scrambled Sobol sequence (scipy.stats.qmc.Sobol, Owen scrambling, seeded);
  its balance properties hold for n a power of two.
* Saltelli: the A, B and AB_i matrices of Saltelli (2002) from the two halves of a 2k-dimensional scrambled Sobol
  sequence, N(k + 2) runs for N base samples, for the Sobol indices.
A categorical or bool factor in a sampled design takes level ⌊u·L⌋ of its L levels at the unit coordinate u.

Analysis
--------
* Sobol indices (Sobol 1993) with the Jansen (1999) estimators as Saltelli et al. (2010, Table 2, (f)) give them:
  first order S_i = (V − ½·mean((f(B) − f(AB_i))²)) / V, total S_Ti = ½·mean((f(A) − f(AB_i))²) / V, V the variance
  of f(A) and f(B) together; percentile bootstrap intervals (Efron and Tibshirani 1993) over the N base rows.
* Main effects (factorial designs; binned for the sampled designs): the mean response at each level of a factor with
  a t interval, and the effect: the mean at its last level minus the mean at its first (the classical two-level main
  effect), with a Welch t interval that treats the other factors' variation as noise. In a Latin hypercube, a Sobol
  sequence or the A and B rows of a Saltelli design, a continuous factor's values are cut into quantile bins (six by
  default, at most one per two runs) and the effect is the last bin's mean minus the first's.
* The best run: the run with the best value of the objective response (the first response unless set) in that
  metric's better direction, against the rig as it is.

Runs are evaluated by core.metrics.evaluate on the project with the run's factors set (core.factors.apply, a new
project each run), through core.jobs.run_study and its ResultCache, so a repeated point (a cancelled run started
again, two responses of one study) is not computed twice. Runs are serial by default; workers > 1 evaluates them in a
pool of that many processes (spawned: each process imports DexVantage and computes its own studies; results are
identical, the order of evaluation is not). A run is cancellable between points; a cancelled experiment keeps the runs
done and analyses what they allow.

Known limits
------------
Every response is the study's own figure, with that study's assumptions and limits. Sobol indices assume independent
factors and settle slowly: read them with their intervals (a few hundredths wide before small differences mean
anything); bootstrap percentile intervals over quasi-random rows are conservative. Main-effect intervals treat the
other factors as noise and say nothing about interactions; a resolution III fraction aliases main effects with
two-factor interactions. A run where a response cannot be evaluated (a study refuses: no robot, a sensor hidden) has
no value there and is left out of that response's analysis, with the count reported. Locked sensors are not varied
(the run is refused, said per run). A process pool pays the import of DexVantage and the robot models once per
process and is only worth it for studies slower than about a second per run; it does not work in a frozen bundle
without multiprocessing's freeze support.
"""
from __future__ import annotations

import copy
import csv
import io
import itertools
import json
import math
import os
import time
import traceback
import warnings
from dataclasses import dataclass, field

import numpy as np

from . import factors as F
from . import jobs
from . import metrics as M

SCHEMA = 'dexvantage.experiment/1'                 # design.json
STUDY_ID = 'experiments'
METHODS = ('full_factorial', 'fractional', 'lhs', 'sobol', 'saltelli')
METHOD_LABEL = {'full_factorial': 'Full factorial', 'fractional': 'Fractional factorial (two levels)',
                'lhs': 'Latin hypercube', 'sobol': 'Sobol sequence', 'saltelli': 'Sobol indices (Saltelli sampling)'}
KINDS = ('continuous', 'categorical', 'bool')
RESOLUTIONS = ('III', 'IV')
ROMAN = {3: 'III', 4: 'IV', 5: 'V', 6: 'VI', 7: 'VII', 8: 'VIII'}
DEFAULTS = {'method': 'lhs', 'n': 32, 'levels': 2, 'resolution': 'IV', 'seed': 0, 'bootstrap': 500,
            'confidence': 0.95, 'workers': 0, 'bins': 6}
LIMITS = {'n': (2, 100_000), 'levels': (2, 11), 'seed': (0, 2 ** 31 - 1), 'bootstrap': (0, 5000),
          'confidence': (0.5, 0.999), 'workers': (0, 32), 'bins': (2, 12)}
MAX_FACTORS = 32
MAX_LEVELS = 64
MAX_RUNS = 20_000                 # runs of a project experiment (each one evaluates studies)
MAX_POINTS = 1_000_000            # points of a function experiment (run_function: the methods' own checks)
CACHE_SIZE = 8                    # whole study results kept between runs (core.jobs.ResultCache; a coverage result
                                  # is about 1.7 MB of JSON on the UR5e sample, so few)
ROW_CACHE_SIZE = 20_000           # runs' response values kept between experiments (RowCache: a few hundred bytes each)
# the default range offered for a quantity picked from the rig: a DexVantage default, not a standard
DEFAULT_SPAN = {'position_m': 0.02, 'angle_deg': 5.0, 'setting_rel': 0.25}
REFERENCES = (
    'I. M. Sobol\', "Sensitivity estimates for nonlinear mathematical models", Mathematical Modelling and '
    'Computational Experiments 1(4), 407–414, 1993.',
    'A. Saltelli, "Making best use of model evaluations to compute sensitivity indices", Computer Physics '
    'Communications 145(2), 280–297, 2002.',
    'A. Saltelli, P. Annoni, I. Azzini, F. Campolongo, M. Ratto and S. Tarantola, "Variance based sensitivity analysis '
    'of model output. Design and estimator for the total sensitivity index", Computer Physics Communications 181(2), '
    '259–270, 2010.',
    'M. J. W. Jansen, "Analysis of variance designs for model output", Computer Physics Communications 117(1–2), '
    '35–43, 1999.',
    'M. D. McKay, R. J. Beckman and W. J. Conover, "A comparison of three methods for selecting values of input '
    'variables in the analysis of output from a computer code", Technometrics 21(2), 239–245, 1979.',
    'G. E. P. Box, J. S. Hunter and W. G. Hunter, Statistics for Experimenters, 2nd ed., Wiley, 2005 (fractional '
    'factorials, resolution, chapters 6 and 7).',
    'B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap, Chapman & Hall, 1993.',
    'SciPy, scipy.stats.qmc (Sobol, discrepancy), https://docs.scipy.org/doc/scipy/reference/stats.qmc.html.',
)
KNOWN_LIMITS = ('Every response is the study\'s own figure, with that study\'s assumptions and limits. Sobol indices '
                'assume independent factors; read them with their intervals. Main-effect intervals treat the other '
                'factors as noise and ignore interactions; a resolution III fraction aliases main effects with '
                'two-factor interactions. Runs a study refuses are left out of that response\'s analysis (counted). '
                'Locked sensors are not varied.')


class ExperimentError(jobs.StudyError):
    """A design that cannot be run; the message says why in plain words."""


def _num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _int_in(v, key, default):
    lo, hi = LIMITS[key]
    try:
        x = int(round(float(v)))
    except (TypeError, ValueError, OverflowError):
        return default
    return min(max(x, lo), hi)


def _qmc_sobol(d: int, seed: int):
    from scipy.stats import qmc
    try:
        return qmc.Sobol(d, scramble=True, rng=int(seed))
    except TypeError:                                   # scipy before 1.15 names it seed
        return qmc.Sobol(d, scramble=True, seed=int(seed))


# =========================================================================== factors
@dataclass
class Factor:
    path: str
    kind: str = 'continuous'           # continuous | categorical | bool
    lo: float = 0.0
    hi: float = 1.0
    levels: list = field(default_factory=list)       # categorical: the levels in order; bool: [False, True]
    n_levels: int = 0                  # continuous, full factorial: evenly spaced levels (0 = the design's default)
    integer: bool = False              # continuous: round to whole numbers
    name: str = ''                     # the user's name for it ('' = from describe())

    def to_dict(self) -> dict:
        d = {'path': self.path, 'kind': self.kind}
        if self.kind == 'continuous':
            d.update(lo=self.lo, hi=self.hi)
            if self.n_levels:
                d['n_levels'] = self.n_levels
            if self.integer:
                d['integer'] = True
        elif self.kind == 'categorical':
            d['levels'] = copy.deepcopy(self.levels)
        if self.name:
            d['name'] = self.name
        return d

    @staticmethod
    def from_dict(d, project=None) -> 'Factor':
        """A factor from design.json (raises ExperimentError with a plain sentence for one that cannot be used).
        The kind comes from the entry ('kind'), else from its levels, else from core.factors.describe."""
        if not isinstance(d, dict):
            raise ExperimentError('A factor must be an object with a "path".')
        path = str(d.get('path') or '').strip()
        if not path:
            raise ExperimentError('A factor has no "path" (for example "sensor:Wrist camera.pitch").')
        kind = d.get('kind')
        if kind not in KINDS:
            if isinstance(d.get('levels'), list):
                kind = 'categorical'
            else:
                try:
                    k = F.describe(path, project)['kind'] if ':' in path else 'continuous'
                except jobs.StudyError:
                    k = 'continuous'
                kind = k if k in KINDS else ('categorical' if k in ('list',) else 'continuous')
        f = Factor(path, kind, name=str(d.get('name') or '').strip()[:60])
        if kind == 'bool':
            f.levels = [False, True]
            f.lo, f.hi = 0.0, 1.0
            return f
        if kind == 'categorical':
            lv = d.get('levels')
            if not isinstance(lv, list) or not lv:
                raise ExperimentError(f'The factor {path} is categorical but has no "levels" list.')
            seen = []
            for x in lv:
                if x not in seen:
                    seen.append(copy.deepcopy(x))
            if len(seen) > MAX_LEVELS:
                raise ExperimentError(f'The factor {path} has {len(seen)} levels; at most {MAX_LEVELS}.')
            f.levels = seen
            f.lo, f.hi = 0.0, float(len(seen) - 1)
            return f
        lo, hi = _num(d.get('lo')), _num(d.get('hi'))
        if lo is None or hi is None:
            raise ExperimentError(f'The factor {path} needs a range: "lo" and "hi" as numbers (or "levels" for a '
                                  'categorical factor).')
        f.lo, f.hi = (lo, hi) if lo <= hi else (hi, lo)
        nl = d.get('n_levels')
        f.n_levels = 0 if nl in (None, 0, '') else _int_in(nl, 'levels', 0)
        f.integer = bool(d.get('integer', False))
        return f

    def level_count(self, default: int) -> int:
        if self.kind in ('categorical', 'bool'):
            return len(self.levels)
        if self.hi <= self.lo:
            return 1
        return self.n_levels or default

    def values(self, u) -> list:
        """The factor's values at unit coordinates u (a column in [0, 1])."""
        u = np.clip(np.asarray(u, float), 0.0, 1.0)
        if self.kind in ('categorical', 'bool'):
            L = len(self.levels)
            idx = np.minimum((u * L + 1e-9).astype(int), L - 1)
            return [copy.deepcopy(self.levels[i]) for i in idx]
        x = self.lo + u * (self.hi - self.lo)
        if self.integer:
            return [int(round(v)) for v in x]
        return [float(v) for v in x]

    def code(self, v) -> float:
        """A number for a value: continuous as is, a level its index (main effects, the CSV's order)."""
        if self.kind in ('categorical', 'bool'):
            try:
                return float(self.levels.index(v))
            except ValueError:
                return float('nan')
        x = _num(v)
        return float('nan') if x is None else x


def factor_title(f: Factor, project=None) -> str:
    if f.name:
        return f.name
    try:
        return F.describe(f.path, project)['title'] if ':' in f.path else f.path
    except jobs.StudyError:
        return f.path


def factor_unit(f: Factor) -> str:
    try:
        return F.describe(f.path)['unit'] if ':' in f.path and f.kind == 'continuous' else ''
    except jobs.StudyError:
        return ''


# =========================================================================== the design (design.json)
@dataclass
class Design:
    factors: list = field(default_factory=list)
    responses: list = field(default_factory=list)       # metric ids (run_function: response names)
    method: str = DEFAULTS['method']
    n: int = DEFAULTS['n']                               # lhs / sobol points; saltelli base samples N
    levels: int = DEFAULTS['levels']                     # full factorial: levels of a continuous factor
    resolution: str = DEFAULTS['resolution']             # fractional: III | IV
    seed: int = DEFAULTS['seed']
    bootstrap: int = DEFAULTS['bootstrap']
    confidence: float = DEFAULTS['confidence']
    bins: int = DEFAULTS['bins']                         # binned main effects of the sampled designs
    workers: int = DEFAULTS['workers']                   # 0 or 1: serial (the default); > 1: a process pool
    objective: str = ''                                  # the response that picks the best run ('' = the first)
    variant: str = ''                                    # a variant's uid ('' = the active one)
    name: str = ''

    def to_dict(self) -> dict:
        return {'schema': SCHEMA, 'name': self.name, 'variant': self.variant,
                'factors': [f.to_dict() for f in self.factors], 'responses': list(self.responses),
                'objective': self.objective,
                'design': {'method': self.method, 'n': self.n, 'levels': self.levels, 'resolution': self.resolution,
                           'seed': self.seed},
                'analysis': {'bootstrap': self.bootstrap, 'confidence': self.confidence, 'bins': self.bins},
                'execution': {'workers': self.workers}}

    @staticmethod
    def from_dict(d, project=None) -> 'Design':
        """design.json (or the experiment saved in a project) as a Design; ExperimentError for one that cannot be
        read. Numbers out of range are clamped; a missing setting takes its default."""
        if not isinstance(d, dict):
            raise ExperimentError('The design must be a JSON object.')
        sch = d.get('schema')
        if sch not in (None, SCHEMA):
            raise ExperimentError(f'The design says schema "{sch}"; this version reads "{SCHEMA}".')
        des = d.get('design') if isinstance(d.get('design'), dict) else {}
        ana = d.get('analysis') if isinstance(d.get('analysis'), dict) else {}
        exe = d.get('execution') if isinstance(d.get('execution'), dict) else {}
        flat = {**exe, **ana, **des}
        for k in DEFAULTS:                                # a flat layout ({"method": "lhs", ...}) is read too
            if k in d and k not in flat:
                flat[k] = d[k]
        x = Design()
        raw = d.get('factors') or []
        if not isinstance(raw, list):
            raise ExperimentError('"factors" must be a list.')
        if len(raw) > MAX_FACTORS:
            raise ExperimentError(f'The design has {len(raw)} factors; at most {MAX_FACTORS}.')
        x.factors = [Factor.from_dict(f, project) for f in raw]
        paths = [f.path for f in x.factors]
        dup = sorted({p for p in paths if paths.count(p) > 1})
        if dup:
            raise ExperimentError(f'The factor {dup[0]} is listed twice.')
        rs = d.get('responses') or []
        if not isinstance(rs, list):
            raise ExperimentError('"responses" must be a list of metric ids.')
        x.responses = []
        for r in rs:
            r = r.get('metric') if isinstance(r, dict) else r
            if r and str(r) not in x.responses:
                x.responses.append(str(r))
        m = flat.get('method', DEFAULTS['method'])
        if m not in METHODS:
            raise ExperimentError(f'Unknown design method "{m}"; one of: {", ".join(METHODS)}.')
        x.method = m
        x.n = _int_in(flat.get('n', DEFAULTS['n']), 'n', DEFAULTS['n'])
        x.levels = _int_in(flat.get('levels', DEFAULTS['levels']), 'levels', DEFAULTS['levels'])
        r = str(flat.get('resolution', DEFAULTS['resolution'])).upper()
        if r not in RESOLUTIONS:
            raise ExperimentError(f'Resolution "{r}" is not offered; III or IV.')
        x.resolution = r
        x.seed = _int_in(flat.get('seed', 0), 'seed', 0)
        x.bootstrap = _int_in(flat.get('bootstrap', DEFAULTS['bootstrap']), 'bootstrap', DEFAULTS['bootstrap'])
        c = _num(flat.get('confidence', DEFAULTS['confidence']))
        x.confidence = DEFAULTS['confidence'] if c is None else min(max(c, LIMITS['confidence'][0]),
                                                                    LIMITS['confidence'][1])
        x.bins = _int_in(flat.get('bins', DEFAULTS['bins']), 'bins', DEFAULTS['bins'])
        x.workers = _int_in(flat.get('workers', 0), 'workers', 0)
        x.objective = str(d.get('objective') or '')
        x.variant = str(d.get('variant') or '')
        x.name = str(d.get('name') or '')[:120]
        return x


def load_design(src, project=None) -> Design:
    """A Design from a design.json path, JSON text or a dict."""
    if isinstance(src, Design):
        return src
    if isinstance(src, dict):
        return Design.from_dict(src, project)
    text = str(src)
    if not text.lstrip().startswith('{'):
        try:
            with open(text, encoding='utf-8') as fh:
                text = fh.read()
        except OSError as e:
            raise ExperimentError(f'The design {os.path.basename(text)} could not be read: {e.strerror or e}.') \
                from None
    try:
        doc = json.loads(text)
    except ValueError as e:
        raise ExperimentError(f'The design is not valid JSON ({e}).') from None
    return Design.from_dict(doc, project)


def settings_of(project) -> dict:
    """The experiment saved in the project (project.analysis.studies['experiments']), as stored."""
    return copy.deepcopy(project.analysis.studies.get(STUDY_ID) or {})


# =========================================================================== plans (the design's points)
@dataclass
class Plan:
    method: str
    U: np.ndarray                      # (n, k) unit coordinates
    info: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.U.shape[0])

    def block(self, i: int) -> str:
        """Saltelli: 'A', 'B' or 'AB<i>' (factor i from B, 1-based); '' for the other designs."""
        if self.method != 'saltelli':
            return ''
        b = i // self.info['N']
        return 'A' if b == 0 else 'B' if b == 1 else f'AB{b - 1}'


def _fractional_k(k: int, resolution: str) -> int:
    """The number of base factors m of the smallest two-level fraction for k factors at the resolution."""
    m = 1
    if resolution == 'III':
        while 2 ** m < k + 1:
            m += 1
    else:
        while 2 ** (m - 1) < k:
            m += 1
    return min(m, k)


def fractional_signs(k: int, resolution: str = 'IV') -> tuple[np.ndarray, list]:
    """The ±1 matrix (2^m, k) of a two-level fraction for k factors and its generators ('E=ABC'): the first m
    columns a full factorial in standard order (the first factor alternating fastest), the others interaction columns
    of the base factors (module docstring)."""
    if k < 1:
        raise ExperimentError('A fractional factorial needs at least one factor.')
    if resolution not in RESOLUTIONS:
        raise ExperimentError(f'Resolution "{resolution}" is not offered; III or IV.')
    m = _fractional_k(k, resolution)
    n = 2 ** m
    base = np.array([[1.0 if (r >> j) & 1 else -1.0 for j in range(m)] for r in range(n)])
    if resolution == 'III':
        subsets = [c for s in range(2, m + 1) for c in itertools.combinations(range(m), s)]
    else:
        subsets = [c for s in range(m - (1 - m % 2), 2, -2) for c in itertools.combinations(range(m), s)]
        subsets = sorted(subsets, key=lambda c: (-len(c), c)) if subsets else []
    letters = 'ABCDEFGHJKLMNPQRSTUVWXYZ'
    name = (lambda j: letters[j] if j < len(letters) else f'X{j + 1}')                  # noqa: E731
    cols, gens = [base[:, j] for j in range(m)], []
    for j, c in zip(range(m, k), subsets):
        cols.append(np.prod(base[:, list(c)], axis=1))
        gens.append(f'{name(j)}={"".join(name(i) for i in c)}')
    if len(cols) < k:                                    # cannot happen for the counts above; said plainly if it did
        raise ExperimentError(f'No two-level resolution {resolution} fraction of {n} runs holds {k} factors.')
    return np.stack(cols, axis=1), gens


def resolution_of(S: np.ndarray, max_len: int = 8) -> int | None:
    """The resolution of a two-level design found by brute force: the length of the shortest set of its columns
    whose elementwise product is constant (a word of the defining relation); None when no word up to max_len exists
    (a full factorial)."""
    S = np.asarray(S, float)
    k = S.shape[1]
    for L in range(1, min(k, max_len) + 1):
        for c in itertools.combinations(range(k), L):
            p = np.prod(S[:, list(c)], axis=1)
            if np.all(p == p[0]):
                return L
    return None


def point_count(design: Design) -> int:
    k = len(design.factors)
    if k == 0:
        return 0
    m = design.method
    if m == 'full_factorial':
        return int(np.prod([max(f.level_count(design.levels), 1) for f in design.factors], dtype=float))
    if m == 'fractional':
        return 2 ** _fractional_k(k, design.resolution)
    if m == 'saltelli':
        return design.n * (k + 2)
    return design.n


def design_text(design: Design) -> str:
    """'1 536 runs = 256 × (4 + 2)': how the count comes about."""
    k, n, m = len(design.factors), point_count(design), design.method
    if not k:
        return 'No factor yet'
    if m == 'full_factorial':
        return f'{n:,} runs = ' + ' × '.join(str(f.level_count(design.levels)) for f in design.factors)
    if m == 'fractional':
        mm = _fractional_k(k, design.resolution)
        return f'{n:,} runs = 2^({k} − {k - mm})' if mm < k else f'{n:,} runs = 2^{k} (the full factorial)'
    if m == 'saltelli':
        return f'{n:,} runs = {design.n} × ({k} + 2)'
    return f'{n:,} runs'


def make_plan(design: Design, limit: int = MAX_RUNS) -> Plan:
    """The design's points in unit coordinates; ExperimentError when it has no factor or too many runs."""
    fs = design.factors
    k = len(fs)
    if k == 0:
        raise ExperimentError('The experiment has no factor: add one (pick a quantity of the rig).')
    n = point_count(design)
    if n > limit:
        raise ExperimentError(f'The design has {n:,} runs, more than {limit:,}: use fewer levels, fewer factors or a '
                              'smaller n.')
    m = design.method
    if m == 'full_factorial':
        axes = [np.linspace(0.0, 1.0, f.level_count(design.levels)) if f.level_count(design.levels) > 1
                else np.zeros(1) for f in fs]
        U = np.array(list(itertools.product(*axes[::-1])), float)[:, ::-1].reshape(-1, k)   # first factor fastest
        return Plan(m, U, {'levels': [len(a) for a in axes]})
    if m == 'fractional':
        bad = [f.path for f in fs if f.level_count(2) != 2]
        if bad:
            raise ExperimentError(f'A two-level fraction needs every factor at two levels: {bad[0]} has '
                                  f'{fs[[f.path for f in fs].index(bad[0])].level_count(2)}. Use the full factorial '
                                  'or a sampled design for it.')
        S, gens = fractional_signs(k, design.resolution)
        res = resolution_of(S)
        return Plan(m, (S + 1.0) / 2.0, {'generators': gens, 'resolution': res,
                                        'resolution_text': ROMAN.get(res, 'full') if res else 'full factorial',
                                        'requested': design.resolution, 'signs': S.astype(int).tolist()})
    rng = np.random.default_rng(design.seed)
    if m == 'lhs':
        U = np.empty((n, k))
        for j in range(k):
            U[:, j] = (rng.permutation(n) + rng.random(n)) / n
        return Plan(m, U, {})
    if m == 'sobol':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')               # n not a power of two: less balanced, still valid
            U = _qmc_sobol(k, design.seed).random(n)
        return Plan(m, U, {'power_of_two': bool(n & (n - 1) == 0)})
    if m == 'saltelli':
        N = design.n
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            S = _qmc_sobol(2 * k, design.seed).random(N)
        A, B = S[:, :k], S[:, k:]
        blocks = [A, B]
        for i in range(k):
            ABi = A.copy()
            ABi[:, i] = B[:, i]
            blocks.append(ABi)
        return Plan(m, np.vstack(blocks), {'N': N, 'k': k, 'power_of_two': bool(N & (N - 1) == 0)})
    raise ExperimentError(f'Unknown design method "{m}".')


def plan_rows(plan: Plan, fs) -> list:
    """The plan's points as factor values: a list of {path: value}, one per run."""
    cols = [f.values(plan.U[:, j]) for j, f in enumerate(fs)]
    return [{f.path: c[i] for f, c in zip(fs, cols)} for i in range(plan.n)]


def centred_l2_discrepancy(U) -> float:
    """Hickernell's (1998) centred L2 discrepancy of points U in [0, 1]^d, from his closed form written out:
    CD² = (13/12)^d − (2/n) Σ_i Π_k (1 + ½|z_ik| − ½ z_ik²) + (1/n²) Σ_ij Π_k (1 + ½|z_ik| + ½|z_jk| − ½|x_ik − x_jk|),
    z = x − ½. Returns CD² (scipy.stats.qmc.discrepancy(method='CD') returns the same squared figure)."""
    X = np.asarray(U, float)
    n, d = X.shape
    z = np.abs(X - 0.5)
    t1 = (13.0 / 12.0) ** d
    t2 = 2.0 / n * np.sum(np.prod(1 + 0.5 * z - 0.5 * z * z, axis=1))
    D = np.abs(X[:, None, :] - X[None, :, :])
    t3 = np.sum(np.prod(1 + 0.5 * z[:, None, :] + 0.5 * z[None, :, :] - 0.5 * D, axis=2)) / (n * n)
    return float(t1 - t2 + t3)


def iid_discrepancy(n: int, d: int) -> float:
    """The expected CD² of n independent uniform points in d dimensions: ((5/4)^d − (13/12)^d) / n (the kernel's
    diagonal mean minus its off-diagonal mean, divided by n)."""
    return ((5.0 / 4.0) ** d - (13.0 / 12.0) ** d) / n


# =========================================================================== estimators
def _pct_ci(samples, conf):
    a = (1.0 - conf) / 2.0
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    if not len(s):
        return [None, None]
    return [float(np.quantile(s, a)), float(np.quantile(s, 1.0 - a))]


def sobol_indices(fA, fB, fAB, bootstrap=500, confidence=0.95, seed=0) -> dict:
    """First-order and total Sobol indices with the Jansen (1999) estimators (Saltelli et al. 2010, Table 2 (f)) from
    f(A), f(B) (N,) and f(AB_i) (k, N), with percentile bootstrap intervals over the N rows. Rows with a missing value
    anywhere are left out (counted in 'dropped')."""
    fA, fB, fAB = np.asarray(fA, float), np.asarray(fB, float), np.atleast_2d(np.asarray(fAB, float))
    k = fAB.shape[0]
    ok = np.isfinite(fA) & np.isfinite(fB) & np.isfinite(fAB).all(axis=0)
    fA, fB, fAB = fA[ok], fB[ok], fAB[:, ok]
    N = len(fA)
    out = {'N': int(N), 'dropped': int((~ok).sum()), 'S1': [None] * k, 'ST': [None] * k,
           'S1_ci': [[None, None] for _ in range(k)], 'ST_ci': [[None, None] for _ in range(k)], 'V': None,
           'mean': None, 'confidence': confidence, 'bootstrap': int(bootstrap),
           'estimator': 'Jansen (1999): S1 = (V − ½·mean((f(B) − f(AB_i))²)) / V, ST = ½·mean((f(A) − f(AB_i))²) / V'}
    if N < 2:
        out['note'] = 'Too few runs with every value to estimate the indices.'
        return out

    def est(a, b, ab):
        V = np.var(np.concatenate([a, b], axis=-1), axis=-1, ddof=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            s1 = (V[..., None] - 0.5 * np.mean((b[..., None, :] - ab) ** 2, axis=-1)) / V[..., None]
            st = 0.5 * np.mean((a[..., None, :] - ab) ** 2, axis=-1) / V[..., None]
        return s1, st, V
    s1, st, V = est(fA, fB, fAB)
    out['V'] = float(V)
    out['mean'] = float(np.mean(np.concatenate([fA, fB])))
    if not V > 0:
        out['note'] = 'The response does not vary over the design: no index.'
        return out
    out['S1'], out['ST'] = [float(x) for x in s1], [float(x) for x in st]
    B = int(bootstrap)
    if B > 0:
        rng = np.random.default_rng(seed)
        S1b, STb = [], []
        chunk = max(1, int(4_000_000 // max(N * (k + 2), 1)))
        done = 0
        while done < B:
            c = min(chunk, B - done)
            idx = rng.integers(0, N, size=(c, N))
            a, b, ab = fA[idx], fB[idx], np.moveaxis(fAB[:, idx], 0, 1)        # (c, N), (c, N), (c, k, N)
            s1b, stb, _ = est(a, b, ab)
            S1b.append(s1b)
            STb.append(stb)
            done += c
        S1b, STb = np.concatenate(S1b), np.concatenate(STb)
        out['S1_ci'] = [_pct_ci(S1b[:, i], confidence) for i in range(k)]
        out['ST_ci'] = [_pct_ci(STb[:, i], confidence) for i in range(k)]
    return out


def _t_half(sd, n, conf):
    if n < 2 or sd is None or not np.isfinite(sd):
        return None
    from scipy.stats import t as _t
    return float(_t.ppf(0.5 + conf / 2.0, n - 1) * sd / math.sqrt(n))


def _groups(x, categorical, bins, sampled=False):
    """Group codes for a factor column: its distinct values (categorical, or a factorial's few levels), else quantile
    bins (always for a continuous factor of a sampled design, at most one bin per two points). Returns (codes (n,)
    int with −1 for missing, centres, edges or None)."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    vals = np.unique(x[ok])
    if sampled and not categorical:
        bins = max(2, min(bins, int(ok.sum()) // 2))
    if categorical or (len(vals) <= 12 and not sampled):
        codes = np.full(len(x), -1)
        codes[ok] = np.searchsorted(vals, x[ok])
        return codes, [float(v) for v in vals], None
    edges = np.unique(np.quantile(x[ok], np.linspace(0, 1, bins + 1)))
    codes = np.full(len(x), -1)
    codes[ok] = np.clip(np.searchsorted(edges, x[ok], side='right') - 1, 0, len(edges) - 2)
    centres = [float(x[ok][codes[ok] == b].mean()) if (codes[ok] == b).any() else float('nan')
               for b in range(len(edges) - 1)]
    return codes, centres, [float(e) for e in edges]


def welch(a, b, confidence=0.95):
    """mean(b) − mean(a) with a Welch t interval: (difference, [low, high]) ([None, None] with fewer than two values
    on a side; a zero-width interval when both sides are constant)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return None, [None, None]
    d = float(b.mean() - a.mean())
    if len(a) < 2 or len(b) < 2:
        return d, [None, None]
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    if se == 0:
        return d, [d, d]
    df = (va + vb) ** 2 / (va * va / (len(a) - 1) + vb * vb / (len(b) - 1))
    from scipy.stats import t as _t
    h = float(_t.ppf(0.5 + confidence / 2.0, df) * se)
    return d, [d - h, d + h]


def main_effects(X, y, fs, confidence=0.95, bins=6, sampled=False) -> list:
    """Per factor: the mean response per level (or, sampled=True for a sampled design's continuous factor, per
    quantile bin of its values) with a t interval and the count, and the effect: the mean at the last level (bin)
    minus the mean at the first with a Welch interval. X: (n, k) factor codes (Factor.code)."""
    out = []
    X, y = np.asarray(X, float), np.asarray(y, float)
    for j, f in enumerate(fs):
        codes, centres, edges = _groups(X[:, j], f.kind != 'continuous', bins, sampled)
        rows = []
        for b, c in enumerate(centres):
            sel = (codes == b) & np.isfinite(y)
            n = int(sel.sum())
            m = float(y[sel].mean()) if n else None
            sd = float(y[sel].std(ddof=1)) if n > 1 else None
            h = _t_half(sd, n, confidence)
            lev = f.levels[int(c)] if f.kind != 'continuous' and 0 <= int(c) < len(f.levels) else None
            rows.append({'x': c, 'level': lev, 'n': n, 'mean': m, 'sd': sd,
                         'lo': None if h is None else m - h, 'hi': None if h is None else m + h})
        eff, ci = (None, [None, None])
        if len(centres) >= 2:
            eff, ci = welch(y[(codes == 0)], y[(codes == len(centres) - 1)], confidence)
        means = [r['mean'] for r in rows if r['mean'] is not None]
        out.append({'factor': f.path, 'binned': edges is not None, 'edges': edges, 'levels': rows,
                    'effect': eff, 'effect_ci': ci, 'range': (max(means) - min(means)) if means else None})
    return out


# =========================================================================== analysis
def analyse(plan: Plan, fs, rows: list, Y: np.ndarray, names: list, design: Design) -> dict:
    """Per response: count, missing, mean, sd, min, max; main effects (factorial and sampled designs); Sobol indices
    (Saltelli)."""
    X = np.array([[f.code(r[f.path]) for f in fs] for r in rows], float).reshape(len(rows), len(fs))
    out = {}
    for c, name in enumerate(names):
        y = Y[:, c] if Y.size else np.zeros(0)
        fin = y[np.isfinite(y)]
        a = {'n': int(len(fin)), 'missing': int(len(y) - len(fin))}
        if len(fin):
            a.update(mean=float(fin.mean()), sd=float(fin.std(ddof=1)) if len(fin) > 1 else 0.0,
                     min=float(fin.min()), max=float(fin.max()))
        if plan.method == 'saltelli':
            N, k = plan.info['N'], plan.info['k']
            a['sobol'] = sobol_indices(y[:N], y[N:2 * N], y[2 * N:].reshape(k, N), design.bootstrap,
                                       design.confidence, design.seed)
            a['sobol']['factors'] = [f.path for f in fs]
            A = slice(0, 2 * N)                           # A and B are independent samples: binned effects from them
            a['main_effects'] = main_effects(X[A], y[A], fs, design.confidence, design.bins, sampled=True)
        elif len(fin):
            a['main_effects'] = main_effects(X, y, fs, design.confidence, design.bins,
                                             sampled=plan.method in ('lhs', 'sobol'))
        out[name] = a
    return out


def _best(Y, col, better):
    y = Y[:, col]
    ok = np.flatnonzero(np.isfinite(y))
    if not len(ok):
        return None
    i = ok[np.argmax(y[ok])] if better == 'higher' else ok[np.argmin(y[ok])]
    return int(i)


# =========================================================================== evaluating runs
def _modules_for(ids) -> list:
    """The modules a spawned process imports so the metrics and their studies are registered there."""
    mods = []
    for mid in ids:
        m = M.METRICS.get(mid)
        if m is None:
            continue
        info = jobs.STUDIES.get(m.study)
        for fn in (m.extract, getattr(info, 'fn', None)):
            mod = getattr(fn, '__module__', None)
            if mod and mod not in mods and mod not in ('__main__', '__mp_main__'):
                mods.append(mod)
    return mods


def _row_failed(ids, status, reason) -> dict:
    return {mid: {'value': None, 'status': status, 'reason': reason} for mid in ids}


class RowCache:
    """A small thread-safe LRU of runs' response values, keyed by the responses and the core.jobs.ResultCache keys
    of their studies on the run's project (its content hash, the study settings, the library entries, a dataset's
    fingerprint): a run already evaluated is not evaluated again, without keeping the studies' whole results."""

    def __init__(self, size=ROW_CACHE_SIZE):
        import threading
        from collections import OrderedDict
        self.size = size
        self._d = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0

    def get(self, key):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                self.hits += 1
                return copy.deepcopy(self._d[key])
            return None

    def put(self, key, value):
        with self._lock:
            self._d[key] = copy.deepcopy(value)
            self._d.move_to_end(key)
            while len(self._d) > self.size:
                self._d.popitem(last=False)


def evaluate_row(base, values: dict, ids, variant: str = '', cache=None, cancel=None, rows=None) -> dict | None:
    """The responses of one run: the project with the run's factors set (core.factors.apply; a new project), its
    metrics through core.metrics.evaluate. Never raises for a refusal: the run's responses say why. None when cancel()
    turned true while it ran (a study stopped part-way gives no value to keep). rows: a RowCache (with cache)."""
    try:
        p = F.apply(base, values, variant=variant or None)
    except jobs.StudyError as e:
        return _row_failed(ids, 'refused', str(e))
    if variant:
        p.active_uid = variant
    key = None
    if rows is not None and cache is not None:
        try:
            studies = sorted({M.METRICS[mid].study for mid in ids if mid in M.METRICS})
            key = (tuple(ids), tuple(cache.key(s, p) for s in studies))
        except Exception:                                 # noqa: BLE001 — no key: evaluated, not cached
            key = None
        hit = rows.get(key) if key is not None else None
        if hit is not None:
            return hit
    try:
        out = M.evaluate(p, ids, cache=cache, ctx=jobs.Ctx(cancel=cancel))
    except jobs.StudyError as e:
        out = _row_failed(ids, 'refused', str(e))
    except Exception as e:                                # noqa: BLE001 — said per run, the traceback logged
        traceback.print_exc()
        first = (str(e).splitlines() or [''])[0][:200]
        return _row_failed(ids, 'error', f'The study stopped with an unexpected error ({type(e).__name__}: {first}).')
    if cancel is not None and cancel():
        return None
    if key is not None:
        rows.put(key, out)
    return out


_WORKER: dict = {}


def _worker_init(base, ids, variant, modules):
    import importlib
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception:                                 # noqa: BLE001 — its metrics then say "no metric called"
            traceback.print_exc()
    M.load_metrics()
    jobs.load_studies()
    _WORKER.update(base=base, ids=list(ids), variant=variant, cache=jobs.ResultCache(CACHE_SIZE))


def _worker_row(i, values):
    w = _WORKER
    return i, evaluate_row(w['base'], values, w['ids'], w['variant'], w['cache'])


def _run_rows(base, rows, ids, variant, ctx, cache, workers, on_row, row_cache=None):
    """Evaluate every row (serial, or in a process pool when workers > 1); on_row(i, result) as each finishes.
    Returns the count done; stops early when ctx is cancelled."""
    n = len(rows)
    done = 0
    if workers <= 1 or n < 2:
        for i, values in enumerate(rows):
            if ctx.cancelled():
                break
            res = evaluate_row(base, values, ids, variant, cache, ctx.cancelled, row_cache)
            if res is None:
                break
            on_row(i, res)
            done += 1
            ctx.report(done / n, f'run {done} of {n}')
        return done
    import multiprocessing as mp
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    ex = ProcessPoolExecutor(max_workers=min(workers, n), mp_context=mp.get_context('spawn'),
                             initializer=_worker_init, initargs=(base, ids, variant, _modules_for(ids)))
    try:
        pending, nxt = set(), 0
        window = 2 * workers                              # a few queued per process: a cancel waits for them only
        while nxt < n or pending:
            while nxt < n and len(pending) < window and not ctx.cancelled():
                pending.add(ex.submit(_worker_row, nxt, rows[nxt]))
                nxt += 1
            if not pending:
                break
            fin, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
            for fu in fin:
                i, res = fu.result()
                on_row(i, res)
                done += 1
                ctx.report(done / n, f'run {done} of {n} ({workers} processes)')
            if ctx.cancelled():
                for fu in pending:
                    fu.cancel()
                break
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    return done


def check_design(project, design: Design) -> list:
    """Plain sentences for what stops the design running on this project (empty: it can run)."""
    out = []
    if not design.factors:
        out.append('The experiment has no factor: add one (pick a quantity of the rig).')
    if not design.responses:
        out.append('The experiment has no response: pick at least one figure to measure.')
    M.load_metrics()
    for mid in design.responses:
        if mid not in M.METRICS:
            out.append(f'There is no metric called {mid}.')
        elif M.METRICS[mid].study == STUDY_ID:
            out.append(f'{mid} is a figure of the experiment itself; it cannot be a response.')
    if design.variant and project.variant_by_uid(design.variant) is None:
        out.append(f'The project has no variant {design.variant}.')
    for f in design.factors:
        try:
            F.get(project, f.path, variant=design.variant or None)
            d = F.describe(f.path, project, variant=design.variant or None)
        except jobs.StudyError as e:
            out.append(str(e))
            continue
        if d.get('locked'):
            out.append(f'{d["title"]}: the sensor is locked; unlock it to vary it.')
        elif f.path.startswith('json:') and F.locked_target(project, f.path) is not None:
            out.append(f'{f.path} reaches into the locked sensor {F.locked_target(project, f.path).name}; unlock it '
                       'to vary it.')
    if design.objective and design.objective not in design.responses:
        out.append(f'The objective {design.objective} is not one of the responses.')
    try:
        if design.factors:
            make_plan(design)
    except ExperimentError as e:
        out.append(str(e))
    return out


def run(project, design, ctx: jobs.Ctx | None = None, cache: jobs.ResultCache | None = None,
        rows: RowCache | None = None) -> dict:
    """Run the design on the project: every run's factors, responses and status, the analysis and the best run.
    Raises ExperimentError when the design cannot run (every problem check_design finds, in one message). cache: the
    study results kept between runs (a new one of CACHE_SIZE by default); rows: a RowCache of runs' values kept
    between experiments (none by default; the study keeps one)."""
    ctx = ctx or jobs.Ctx()
    design = load_design(design, project)
    problems = check_design(project, design)
    if problems:
        raise ExperimentError(' '.join(problems[:5]) + (f' ({len(problems) - 5} more.)' if len(problems) > 5 else ''))
    t0 = time.perf_counter()
    cache = cache if cache is not None else jobs.ResultCache(CACHE_SIZE)
    fs, ids = design.factors, list(design.responses)
    plan = make_plan(design)
    runs = plan_rows(plan, fs)
    ctx.report(0.0, 'the rig as it is')
    base = project.snapshot()                    # without the experiment's own settings: no study reads them, and an
    base.analysis.studies.pop(STUDY_ID, None)    # edited design keeps the runs it shares with the old one cached
    base_res = evaluate_row(base, {}, ids, design.variant, cache, ctx.cancelled, rows) or \
        _row_failed(ids, 'not run', 'The experiment was cancelled before the rig as it is was evaluated.')
    Y = np.full((plan.n, len(ids)), np.nan)
    results: list = [None] * plan.n

    def on_row(i, res):
        results[i] = res
        Y[i] = [np.nan if res[mid]['value'] is None else res[mid]['value'] for mid in ids]
    hits0, rhits0 = cache.hits, rows.hits if rows is not None else 0
    workers = design.workers if design.workers > 1 else 0
    done = _run_rows(base, runs, ids, design.variant, ctx, cache, workers, on_row, rows)
    cancelled = done < plan.n
    return _result(project, design, plan, runs, Y, results, base_res, ids, {
        'workers': workers, 'mode': f'process pool of {workers}' if workers else 'serial (the default)',
        'runs_done': done, 'runs_planned': plan.n, 'cache_hits': cache.hits - hits0 if not workers else None,
        'runs_from_cache': (rows.hits - rhits0) if rows is not None and not workers else None,
        'seconds': round(time.perf_counter() - t0, 3)}, cancelled)


def _meta(mid) -> dict:
    m = M.METRICS.get(mid)
    if m is None:
        return {'id': mid, 'title': mid, 'unit': '', 'better': 'higher', 'study': ''}
    return {'id': mid, 'title': m.title, 'unit': m.unit, 'better': m.better, 'study': m.study}


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _result(project, design, plan, rows, Y, results, base_res, ids, execution, cancelled) -> dict:
    fs = design.factors
    resp = [_meta(mid) for mid in ids]
    obj = design.objective or ids[0]
    oc = ids.index(obj)
    better = resp[oc]['better']
    runs = []
    for i, values in enumerate(rows):
        r = results[i]
        runs.append({'run': i + 1, 'block': plan.block(i), 'values': {k: _jsonable(v) for k, v in values.items()},
                     'done': r is not None,
                     'responses': {mid: (None if r is None else _jsonable(r[mid]['value'])) for mid in ids},
                     'status': {mid: ('not run' if r is None else r[mid]['status']) for mid in ids},
                     'reason': {mid: r[mid]['reason'] for mid in ids if r is not None and r[mid].get('reason')}})
    analysis = analyse(plan, fs, rows, Y, ids, design)
    bi = _best(Y, oc, better)
    base_v = base_res[obj]['value']
    best = None
    if bi is not None:
        bv = float(Y[bi, oc])
        improves = base_v is None or (bv > base_v if better == 'higher' else bv < base_v)
        best = {'run': bi + 1, 'response': obj, 'value': bv, 'unit': resp[oc]['unit'], 'better': better,
                'values': runs[bi]['values'], 'baseline': base_v, 'improves': bool(improves),
                'change': None if base_v is None else bv - base_v}
    finfo = []
    for f in fs:
        d = {'path': f.path, 'title': factor_title(f, project), 'unit': factor_unit(f), 'kind': f.kind}
        if f.kind == 'continuous':
            d.update(lo=f.lo, hi=f.hi, integer=f.integer)
            try:
                d['current'] = _jsonable(F.get(project, f.path, variant=design.variant or None))
            except jobs.StudyError:
                d['current'] = None
        else:
            d['levels'] = copy.deepcopy(f.levels)
            try:
                d['current'] = copy.deepcopy(F.get(project, f.path, variant=design.variant or None))
            except jobs.StudyError:
                d['current'] = None
        finfo.append(d)
    info = {k: v for k, v in plan.info.items() if k != 'signs'}
    notes = []
    if plan.method in ('sobol', 'saltelli') and not plan.info.get('power_of_two', True):
        notes.append(f'n = {design.n} is not a power of two: the Sobol sequence\'s balance properties hold only for '
                     'powers of two (the results are still valid).')
    if plan.method == 'fractional':
        notes.append(f'Fractional factorial: generators {", ".join(info["generators"]) or "none"}; resolution '
                     f'{info["resolution_text"]} (found by brute force over the design\'s columns).')
    for mid in ids:                     # Sobol estimates outside [0, 1] or wide intervals: say what they mean
        so = ((analysis.get(mid) or {}).get('sobol') or {})
        est = [x for x in (so.get('S1') or []) + (so.get('ST') or []) if x is not None]
        cis = [c for c in (so.get('S1_ci') or []) + (so.get('ST_ci') or []) if c and None not in c]
        widest = max((c[1] - c[0] for c in cis), default=0.0)
        if est and (min(est) < 0 or max(est) > 1 or widest > 0.5):
            notes.append(f'{mid}: with N = {so.get("N")} base samples the Sobol indices are rough (estimates from '
                         f'{min(est):.2f} to {max(est):.2f}, the widest {so.get("confidence", 0.95) * 100:.0f} % '
                         'interval {:.2f} wide); a true index lies between 0 and 1, so a value outside is sampling '
                         'error. More base samples narrow the intervals.'.format(widest))
    refused = {mid: sum(1 for r in runs if r['done'] and r['status'][mid] != 'ok') for mid in ids}
    for mid, k in refused.items():
        if k:
            why = next((r['reason'][mid] for r in runs if mid in r['reason']), '')
            notes.append(f'{mid}: {k} of {sum(r["done"] for r in runs)} runs have no value' +
                         (f' (for example: {why})' if why else '') + '.')
    return {
        'name': design.name,
        'design': design.to_dict(),
        'method': plan.method, 'method_label': METHOD_LABEL[plan.method], 'design_text': design_text(design),
        'plan': info,
        'factors': finfo,
        'responses': resp,
        'objective': obj,
        'baseline': {mid: {'value': _jsonable(base_res[mid]['value']), 'status': base_res[mid]['status'],
                           **({'reason': base_res[mid]['reason']} if base_res[mid].get('reason') else {})}
                     for mid in ids},
        'runs': runs,
        'analysis': analysis,
        'best': best,
        'headline': _headline(best, resp[oc], len(runs), execution['runs_done'], base_res[obj]),
        'execution': execution,
        'cancelled_at': execution['runs_done'] if cancelled else None,
        'notes': notes,
        'known_limits': KNOWN_LIMITS,
        'references': list(REFERENCES),
    }


def _fmt(v, unit=''):
    if v is None:
        return '—'
    return f'{v:.4g}' + (f' {unit}' if unit else '')


def _headline(best, meta, n, done, base) -> dict:
    title = meta['title']
    if best is None:
        why = base.get('reason') or 'no run gave a value'
        return {'value': None, 'unit': meta['unit'], 'status': 'critical', 'word': 'not evaluated',
                'text': f'No run gave {title.lower()}: {why}'}
    word, sev = ('better than now', 'good') if best['improves'] and best['baseline'] is not None else \
        ('no value now', 'good') if best['baseline'] is None else ('no run beats the rig', 'warning')
    if best['baseline'] is not None and best['change'] == 0:
        word, sev = ('same as now', 'warning')
    text = (f'Best of {done} run{"s" if done != 1 else ""}' + (f' (of {n} planned)' if done < n else '') +
            f': {title.lower()} {_fmt(best["value"], meta["unit"])} at run {best["run"]}, against '
            f'{_fmt(best["baseline"], meta["unit"])} for the rig as it is.')
    if best['improves'] and best['baseline'] is not None and best['change']:
        text += ' Apply sets that run\'s factors (one undo step).'
    return {'value': best['value'], 'unit': meta['unit'], 'status': sev, 'word': word, 'text': text}


def run_function(fn, design, ctx: jobs.Ctx | None = None) -> dict:
    """The same designs and analyses for any function of the factors (the methods' own checks, no project):
    fn({path: value}) -> a number (one response, named by the design's first response or 'y') or {name: number}.
    design.responses name the responses ('better' is 'higher' for all; the best run is reported the same way)."""
    ctx = ctx or jobs.Ctx()
    design = load_design(design)
    names = list(design.responses) or ['y']
    plan = make_plan(design, limit=MAX_POINTS)
    rows = plan_rows(plan, design.factors)
    Y = np.full((plan.n, len(names)), np.nan)
    for i, values in enumerate(rows):
        if ctx.cancelled():
            break
        out = fn(values)
        if not isinstance(out, dict):
            out = {names[0]: out}
        Y[i] = [np.nan if _num(out.get(k)) is None else float(out[k]) for k in names]
    analysis = analyse(plan, design.factors, rows, Y, names, design)
    return {'method': plan.method, 'plan': {k: v for k, v in plan.info.items() if k != 'signs'}, 'U': plan.U,
            'rows': rows, 'Y': Y, 'analysis': analysis, 'responses': names}


# =========================================================================== the rig's quantities (the page's picker)
def candidates(project, variant=None) -> list:
    """Factor templates for the quantities of the rig: [{'path', 'title', 'unit', 'kind', 'group', 'locked',
    'factor': a design.json factor entry}]. Ranges are DexVantage defaults (DEFAULT_SPAN: ±20 mm and ±5° around the
    current value, ±25 % of a numeric setting), to be edited; a model's levels are its current entry and the
    library's other entries of the same kind (up to four)."""
    from . import library as LB
    v = project.variant if variant is None else project.variant_by_uid(variant) or project.variant
    out = []
    pos, ang = DEFAULT_SPAN['position_m'], DEFAULT_SPAN['angle_deg']
    for s in v.sensors:
        group = s.name
        for fld in F.POSITION + F.ANGLES + F.SHIFTS + F.TURNS + ('model', 'enabled'):
            path = f'sensor:{s.uid}.{fld}'
            d = F.describe(path, project, variant=v.uid)
            if fld in F.POSITION + F.ANGLES:
                cur = float(F.get(project, path, variant=v.uid))
                span = pos if fld in F.POSITION else ang
                fac = {'path': path, 'kind': 'continuous', 'lo': round(cur - span, 6), 'hi': round(cur + span, 6)}
            elif fld in F.SHIFTS + F.TURNS:
                span = pos if fld in F.SHIFTS else ang
                fac = {'path': path, 'kind': 'continuous', 'lo': -span, 'hi': span}
            elif fld == 'model':
                try:
                    same = [e.id for e in LB.entries(s.kind) if e.id != s.model_id]
                except Exception:                         # noqa: BLE001 — a library that cannot load: current only
                    same = []
                levels = ([s.model_id] if s.model_id else []) + same[:3]
                if len(levels) < 2:
                    continue
                fac = {'path': path, 'kind': 'categorical', 'levels': levels}
            else:
                fac = {'path': path, 'kind': 'bool'}
            out.append({'path': path, 'title': d['title'], 'unit': d['unit'], 'kind': fac['kind'], 'group': group,
                        'locked': bool(d.get('locked')), 'factor': fac})
    for sid, st in sorted((project.analysis.studies or {}).items()):
        if sid == STUDY_ID or not isinstance(st, dict):
            continue
        for key, val in sorted(st.items()):
            path = f'setting:{sid}.{key}'
            if isinstance(val, bool):
                fac = {'path': path, 'kind': 'bool'}
            elif isinstance(val, (int, float)) and math.isfinite(float(val)) and val != 0:
                r = abs(float(val)) * DEFAULT_SPAN['setting_rel']
                fac = {'path': path, 'kind': 'continuous', 'lo': float(val) - r, 'hi': float(val) + r}
                if isinstance(val, int):
                    fac.update(lo=int(math.floor(val - r)), hi=int(math.ceil(val + r)), integer=True)
            else:
                continue
            out.append({'path': path, 'title': f'{sid} · {key}', 'unit': '', 'kind': fac['kind'],
                        'group': 'Study settings', 'locked': False, 'factor': fac})
    return out


def apply_keys(project, values: dict, variant: str = '') -> list:
    """The project sections a run's factors change (the undo step's keys)."""
    keys = []
    for path in values:
        kind = path.split(':', 1)[0]
        if kind == 'sensor':
            k = f'variant:{variant}' if variant else 'variant'
        elif kind == 'setting':
            k = 'analysis'
        else:
            top = path.split(':', 1)[1].lstrip('/').split('/')[0]
            k = {'name': 'meta', 'notes': 'meta', 'active_uid': 'variants'}.get(top, top)
            from .project import SECTIONS
            if k not in SECTIONS:
                k = 'variants' if top == 'variants' else 'meta'
        if k not in keys:
            keys.append(k)
    return keys


def apply_values(project, values: dict, variant: str = ''):
    """Set the factors on project in place (the page's Apply, inside one store edit): core.factors.apply builds the
    new project; its changed sections replace the project's."""
    q = F.apply(project, values, variant=variant or None)
    for k in apply_keys(project, values, variant):
        key = 'variant:' + (variant or project.active_uid) if k == 'variant' else k
        project.restore(key, copy.deepcopy(q.section(key)))
    return project


# =========================================================================== output
def to_csv(doc: dict) -> str:
    """The runs as CSV: run, block, one column per factor (its path), then per response its value and status."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    fpaths = [f['path'] for f in doc['factors']]
    rids = [r['id'] for r in doc['responses']]
    units = {r['id']: r['unit'] for r in doc['responses']}
    w.writerow(['run', 'block'] + fpaths + [f'{m} ({units[m]})' if units[m] else m for m in rids] +
               [f'{m} status' for m in rids])

    def cell(v):
        if v is None:
            return ''
        if isinstance(v, (list, dict, bool)):
            return json.dumps(v)
        return repr(v) if isinstance(v, float) else str(v)
    for r in doc['runs']:
        w.writerow([r['run'], r['block']] + [cell(r['values'].get(p)) for p in fpaths] +
                   [cell(r['responses'][m]) for m in rids] + [r['status'][m] for m in rids])
    return buf.getvalue()


def write_outputs(doc: dict, project, json_path: str, option: str = 'experiment') -> tuple[str, str]:
    """The result as JSON with provenance and the runs as CSV beside it (same name, .csv); the JSON records the CSV's
    name and sha256. Returns (json path, csv path)."""
    import hashlib

    from . import provenance as PV
    base = json_path[:-5] if json_path.lower().endswith('.json') else json_path
    csv_path = base + '.csv'
    text = to_csv(doc)
    data = text.encode('utf-8')
    doc = dict(doc, files={'csv': os.path.basename(csv_path), 'csv_sha256': hashlib.sha256(data).hexdigest()})
    doc = PV.stamp(doc, project, output=PV.batch_output(doc, json_path, option=option))
    for path, payload in ((csv_path, data),
                          (json_path, (json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False) + '\n')
                           .encode('utf-8'))):
        tmp = path + '.part'
        with open(tmp, 'wb') as fh:
            fh.write(payload)
        os.replace(tmp, path)
    return json_path, csv_path


def default_output(design_path: str) -> str:
    base = design_path[:-5] if design_path.lower().endswith('.json') else design_path
    return base + '.results.json'


def batch(project, design_path: str, out: str | None = None, progress=None) -> tuple[dict, str, str]:
    """--experiment DESIGN.json [--experiment-out OUT.json]: run the design on the project and write the result JSON
    (with provenance) and the runs CSV. Raises ExperimentError (a plain sentence) for a design that cannot run."""
    out = out or default_output(design_path)
    if os.path.abspath(out) == os.path.abspath(design_path):
        raise ExperimentError('The output would overwrite the design file: choose another name with '
                              '--experiment-out.')
    design = load_design(design_path, project)
    doc = run(project, design, jobs.Ctx(progress=progress))
    doc = {'schema': 'dexvantage.experiments/1', 'study': STUDY_ID, **doc}
    try:
        jp, cp = write_outputs(doc, project, out)
    except OSError as e:
        raise ExperimentError(f'Could not write {os.path.basename(out)}: {e.strerror or e}. Choose a folder that '
                              'exists and that you can write to.') from None
    return doc, jp, cp
