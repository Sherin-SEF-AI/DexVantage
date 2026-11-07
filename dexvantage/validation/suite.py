"""The validation suite's machinery: the check registry, the run, and the result.

A check compares one quantity a DexVantage model computes with an independent reference (a closed form written out again
in the check, another library's implementation of the same model, a published table or worked example, a measurement,
a brute-force computation or a Monte Carlo simulation) and reports one error figure against a tolerance: the check
passes when the error is finite and at most the tolerance. A check that raises fails, with the exception named; it
never stops the suite.

Checks are registered with @check (see the checks_* modules) and grouped by the model they validate (MODELS). Each is
reproducible: its random numbers come from a generator seeded with the suite's seed and the check's id, and every
input it uses is either built in the check or the project it was given (said in its detail).

Ported from Vantage validation/suite.py: the same registry, run and result; DexVantage's models.
"""
from __future__ import annotations

import math
import os
import platform
import sys
import tempfile
import time
import traceback
import zlib
from dataclasses import dataclass, field

import numpy as np

SCHEMA = 'dexvantage.validation/1'
DEFAULT_SEED = 20261027

# the models the suite validates, in report order: (id, title, modules, overview anchor). Checks arrive with each
# phase (docs/BUILD_STATE.md); a model without checks yet is left out of the report.
MODELS = (
    ('frames', 'Frames, poses and quaternions', 'dexvantage.core.frames', 'frames'),
    ('kinematics', 'Robot kinematics', 'dexvantage.core.robot', 'kinematics'),
    ('robots', 'Robot catalogue (data consistency)', 'dexvantage.core.robot.catalogue', 'catalogue'),
    ('optics', 'Lens projection and defocus', 'dexvantage.core.optics', 'optics'),
    ('library', 'Hardware library data consistency', 'dexvantage.core.library', 'library'),
    ('occlusion', 'Occlusion by the posed robot', 'dexvantage.core.occlusion', 'occlusion'),
    ('depth', 'Depth validity', 'dexvantage.core.studies.depth_validity', 'depth'),
    ('grasp', 'Grasp self-occlusion', 'dexvantage.core.studies.grasp_visibility', 'grasp'),
    ('groundtruth', 'Rendered against predicted (MuJoCo ground truth)',
     'dexvantage.core.sim, dexvantage.core.studies.groundtruth', 'groundtruth'),
    ('handeye', 'Hand-eye calibration', 'dexvantage.core.calib', 'handeye'),
    ('timing', 'Motion blur, time sync and action-chunk latency',
     'dexvantage.core.studies.motion_blur, dexvantage.core.studies.time_sync, '
     'dexvantage.core.studies.action_chunk_sync', 'timing'),
    ('bandwidth', 'Bandwidth, hosts and power', 'dexvantage.core.studies.bandwidth', 'bandwidth'),
    ('brackets', 'Brackets and mounts', 'dexvantage.core.studies.brackets', 'brackets'),
    ('teleop', 'Teleop overlap and dexterity', 'dexvantage.core.studies.teleop', 'teleop'),
    ('lighting', 'Lighting, glare and IR', 'dexvantage.core.studies.lighting', 'lighting'),
    ('vla', 'VLA tokens, Plücker rays and canonical warps',
     'dexvantage.core.studies.vla_tokens, dexvantage.core.exporters.canonical_warp', 'vla'),
    ('dexhand', 'Multi-finger occlusion and tactile handoff', 'dexvantage.core.studies.dex_hand_tactile', 'dexhand'),
    ('cables', 'Cable flex', 'dexvantage.core.cables, dexvantage.core.studies.cables', 'cables'),
    ('data', 'Dataset readers and the auditor', 'dexvantage.core.datasets, dexvantage.core.audit', 'data'),
    ('tolerance', 'Monte Carlo tolerances and perturbation export',
     'dexvantage.core.tolerance, dexvantage.core.exporters.perturbations', 'tolerance'),
    ('experiments', 'Designed experiments (factorial, Latin hypercube, Sobol)',
     'dexvantage.core.experiments, dexvantage.core.studies.experiments', 'experiments'),
    ('optimise', 'Optimiser and design agent (NSGA-II)', 'dexvantage.core.optimise', 'optimise'),
    ('formats', 'Project and export formats',
     'dexvantage.core.project, dexvantage.core.exporters, dexvantage.core.bridges', 'formats'),
    ('assistant', 'Assistant tools and the MCP server', 'dexvantage.assistant, dexvantage.mcp', 'assistant'),
    ('requirements', 'Requirements scorecard, metrics and reproduction',
     'dexvantage.core.requirements, dexvantage.core.metric_defs, dexvantage.core.reproduce', 'requirements'),
    ('copilot', 'Assistant dock: number check, Copilot gate and edit proposals',
     'dexvantage.assistant.numcheck, dexvantage.assistant.copilot, dexvantage.assistant.edits', 'copilot'),
    ('versioning', 'Project diff and merge, the Git folder form, citations and figures',
     'dexvantage.core.projdiff, dexvantage.core.projfolder, dexvantage.core.cite, dexvantage.core.figures',
     'versioning'),
    ('samples', 'The ALOHA 2, bimanual Panda and G1 samples against their model files',
     'dexvantage.core.robot.bimanual_samples', 'samples'),
    ('workspace', 'Workspace check (task objects inside task volumes)', 'dexvantage.core.workspace_check',
     'workspace'),
    ('robot_io', 'Robot import and export: URDF inertials, USD import, OpenVLA encodings, Isaac principal point',
     'dexvantage.core.exporters.ros2, dexvantage.core.robot.usd_import, dexvantage.core.robot.usd_writer, '
     'dexvantage.core.exporters.lerobot, dexvantage.core.exporters.isaac', 'robot_io'),
    ('campaign', 'Data-collection campaign planner', 'dexvantage.core.campaign', 'campaign'),
    ('publication', 'Publication kit (methods paragraph, LaTeX tables, figures, BibTeX)',
     'dexvantage.core.publication, dexvantage.core.exporters.paper_kit', 'paper_kit'),
    ('tracking', 'Experiment tracker (run store, tables, MLflow layout)', 'dexvantage.core.tracking', 'tracking'),
    ('cells', 'Multi-cell comparison (geometric gap, dataset pooling, drift)',
     'dexvantage.core.cells, dexvantage.core.cells_pool', 'cells'),
    ('stats', 'Statistics and power analysis', 'dexvantage.core.stats, dexvantage.core.statcompare', 'stats'),
    ('cluster', 'Cluster sweeps (process pool, Slurm shards, shared result cache)', 'dexvantage.core.cluster',
     'cluster'),
    ('sim2real', 'Sim-to-real gap (render against recording)', 'dexvantage.core.sim2real', 'sim2real'),
)
MODEL_IDS = tuple(m[0] for m in MODELS)
PLUGIN_MODELS: list = []          # (id, title, modules, anchor) of the models plugins validate (PluginAPI.check)


def all_models() -> tuple:
    """MODELS, then the models the loaded plugins added, in report order."""
    return MODELS + tuple(PLUGIN_MODELS)


def model_ids() -> tuple:
    """Every model id: the built-in ones (MODEL_IDS), then the plugins' ('plugin.<name>')."""
    return tuple(m[0] for m in all_models())
LIMITS = {'seed': (0, 2 ** 31 - 1)}


@dataclass
class ValidationSettings:
    models: tuple = ()             # model ids to run; () = every model
    quick: bool = False            # fewer Monte Carlo trials and samples: a smoke run, tolerances unchanged
    seed: int = DEFAULT_SEED
    cheap_only: bool = False       # only the checks of cost 'cheap' (the self-test's run; no renders of datasets)

    def to_dict(self):
        return {'models': list(self.models), 'quick': self.quick, 'seed': self.seed, 'cheap_only': self.cheap_only}

    @staticmethod
    def from_dict(d) -> 'ValidationSettings':
        """Tolerant of junk: unknown model ids are dropped, the seed clamped."""
        s = ValidationSettings()
        d = d if isinstance(d, dict) else {}
        ms = d.get('models')
        if isinstance(ms, str):
            ms = [m.strip() for m in ms.split(',')]
        if isinstance(ms, (list, tuple)):
            s.models = tuple(m for m in model_ids() if m in {str(x) for x in ms})
        if isinstance(d.get('quick'), bool):
            s.quick = d['quick']
        if isinstance(d.get('cheap_only'), bool):
            s.cheap_only = d['cheap_only']
        try:
            seed = int(d.get('seed', s.seed))
            lo, hi = LIMITS['seed']
            s.seed = min(max(seed, lo), hi)
        except (TypeError, ValueError, OverflowError):
            pass
        return s


@dataclass
class Spec:
    """A registered check."""
    id: str
    model: str
    title: str
    quantity: str             # what is compared, and the error figure reported
    reference: str            # the independent reference it is compared with
    tolerance: float
    unit: str                 # the error figure's unit
    refs: tuple               # keys into references.REFERENCES
    fn: object
    cost: str = 'cheap'       # 'cheap' (< 1 s) | 'study' (seconds)
    tolerance_quick: float | None = None   # the quick run's tolerance when it differs (a band set by the sample size)


@dataclass
class Outcome:
    """What a check function returns: the error figure and what backs it."""
    error: float
    detail: dict = field(default_factory=dict)
    figure: dict | None = None
    ok: bool = True           # a further condition the check requires besides error <= tolerance
    tolerance: float | None = None   # overrides the spec's (a band that depends on the sample size)
    note: str = ''


@dataclass
class Check:
    id: str
    model: str
    title: str
    quantity: str
    reference: str
    refs: tuple
    unit: str
    tolerance: float
    error: float
    passed: bool
    detail: dict
    figure: dict | None
    note: str
    seconds: float
    exception: str = ''


@dataclass
class ValidationResult:
    checks: list
    settings: ValidationSettings
    project: str
    variant: str
    started_utc: str
    seconds: float
    environment: dict
    cancelled: bool = False

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed(self) -> list:
        return [c for c in self.checks if not c.passed]

    @property
    def ok(self) -> bool:
        return bool(self.checks) and not self.failed and not self.cancelled

    def check(self, cid: str) -> Check | None:
        return next((c for c in self.checks if c.id == cid), None)

    def by_model(self) -> list:
        """[(model id, title, modules, anchor, [Check])] in MODELS order, models without checks left out."""
        out = []
        for mid, title, mods, anchor in all_models():
            cs = [c for c in self.checks if c.model == mid]
            if cs:
                out.append((mid, title, mods, anchor, cs))
        return out


CHECKS: list = []


def check(cid, model, title, quantity, reference, tolerance, unit, refs=(), cost='cheap', tolerance_quick=None):
    """Register a check function fn(ctx) -> Outcome. tolerance_quick: the tolerance of a quick run, when the check's
    band depends on its sample size (declared here, so the tolerance listed is the one applied)."""
    assert model in model_ids(), model

    def deco(fn):
        assert not any(s.id == cid for s in CHECKS), cid
        CHECKS.append(Spec(cid, model, title, quantity, reference, float(tolerance), unit, tuple(refs), fn, cost,
                           None if tolerance_quick is None else float(tolerance_quick)))
        return fn
    return deco


def declared_tolerance(spec: Spec, quick: bool = False) -> float:
    """The tolerance a check declares for a full or a quick run."""
    return spec.tolerance_quick if quick and spec.tolerance_quick is not None else spec.tolerance


class Ctx:
    """What a check gets: the project the project-dependent checks look at, an engine, the seed and a scratch folder."""

    def __init__(self, project, engine, variant, settings: ValidationSettings):
        self.settings = settings
        self.quick = settings.quick
        self.given = project is not None
        # the checks never change the project they were given: they look at a copy
        self.project = project.snapshot() if project is not None else self.sample()
        if variant is not None and self.project.variant_by_uid(variant.uid) is not None:
            self.project.active_uid = variant.uid
        self.engine = engine
        self._tmp = None
        self.cache = {}

    def sample(self):
        """The built-in sample project (a fresh copy per call): the bundled SO-100 sample with the UR5e + Robotiq
        2F-85 sample's robot added, so the project-dependent checks see both bundled robots."""
        from ..core.robot.samples import sample_project
        p = sample_project('so100')
        ur = sample_project('ur5e')
        for r in ur.robots:
            r.primary = False
        p.robots += ur.robots
        p.sources.update(ur.sources)
        p.name = 'Validation (built in: the SO-100 and UR5e samples)'
        return p

    def rng(self, cid: str, k: int = 0) -> np.random.Generator:
        """The generator of check cid: seeded with the suite's seed and the check's id, so a check's draws do not
        depend on which other checks ran."""
        return np.random.default_rng([int(self.settings.seed), zlib.crc32(cid.encode('utf-8')), int(k)])

    def tmp(self) -> str:
        if self._tmp is None:
            self._tmp = tempfile.TemporaryDirectory(prefix='dexvantage-validation-')
        return self._tmp.name

    def close(self):
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None


def environment() -> dict:
    """The versions the run used (no paths, nothing of the machine but its platform)."""
    from .. import __version__
    import scipy
    out = {'dexvantage': __version__, 'python': platform.python_version(), 'numpy': np.__version__,
           'scipy': scipy.__version__, 'platform': f'{platform.system()} {platform.machine()}'}
    try:
        import cv2
        out['opencv'] = cv2.__version__
    except Exception:                       # noqa: BLE001 — reported as missing, checks that need it fail
        out['opencv'] = None
    try:
        import mujoco
        out['mujoco'] = mujoco.__version__
    except Exception:                       # noqa: BLE001
        out['mujoco'] = None
    try:
        from PyQt6.QtCore import QT_VERSION_STR
        out['qt'] = QT_VERSION_STR
    except Exception:                       # noqa: BLE001
        out['qt'] = None
    return out


def available_models() -> tuple:
    """The model ids this installation can check (the plugins' models included)."""
    return model_ids()


def load_checks():
    """Import every validation/checks_*.py module once (each registers its checks with @check)."""
    import importlib
    import pkgutil
    for m in pkgutil.iter_modules([os.path.dirname(os.path.abspath(__file__))]):
        if m.name.startswith('checks_'):
            importlib.import_module(f'{__package__}.{m.name}')


def specs(settings: ValidationSettings | None = None) -> list:
    """The registered checks the settings select, in MODELS order (models whose module is not installed left out)."""
    load_checks()
    st = settings or ValidationSettings()
    use = (set(st.models) if st.models else set(model_ids())) & set(available_models())
    order = {m: i for i, m in enumerate(model_ids())}
    return sorted((s for s in CHECKS if s.model in use and (s.cost == 'cheap' or not st.cheap_only)),
                  key=lambda s: order[s.model])


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def run_one(spec: Spec, ctx: Ctx) -> Check:
    t0 = time.perf_counter()
    try:
        out = spec.fn(ctx)
        tol = declared_tolerance(spec, ctx.quick) if out.tolerance is None else float(out.tolerance)
        err = float(out.error)
        passed = _finite(err) and err <= tol and bool(out.ok)
        return Check(spec.id, spec.model, spec.title, spec.quantity, spec.reference, spec.refs, spec.unit, tol, err,
                     passed, out.detail, out.figure, out.note, time.perf_counter() - t0)
    except Exception as e:                  # noqa: BLE001 — a broken model is a failed check, not a crashed suite
        tb = traceback.format_exception(type(e), e, e.__traceback__)
        where = next((ln.strip() for ln in reversed(tb) if 'File "' in ln and 'dexvantage' in ln), '')
        return Check(spec.id, spec.model, spec.title, spec.quantity, spec.reference, spec.refs, spec.unit,
                     declared_tolerance(spec, ctx.quick), float('nan'), False, {}, None, '', time.perf_counter() - t0,
                     exception=f'{type(e).__name__}: {e}'[:400] + (f' ({where[:200]})' if where else ''))


def release_memory(ctx: Ctx | None = None):
    """Drop the process-wide caches the checks of one model filled (compiled robot models, cube maps, lens remap
    plans, rendered board rays, the audit campaign, the checks' own cache) before the next model's checks: a full run
    otherwise grows past 6 GB. Nothing here changes a result; the next user of a cache rebuilds it."""
    import gc
    if ctx is not None:
        ctx.cache.clear()
    for mod, attr in (('dexvantage.core.robot.model', 'clear_cache'), ('dexvantage.core.occlusion', 'CACHE'),
                      ('dexvantage.core.grasps', '_PAD_CACHE'), ('dexvantage.core.calib.synthetic', '_RAYS'),
                      ('dexvantage.core.sim.render', '_PLANS'), ('dexvantage.core.robot.swept', '_CACHE'),
                      ('dexvantage.validation.checks_audit', '_CACHE')):
        obj = getattr(sys.modules.get(mod), attr, None)          # only modules this run imported
        try:
            if hasattr(obj, 'clear'):
                obj.clear()
            elif callable(obj):
                obj()
        except Exception:                   # noqa: BLE001 — freeing memory must never fail a run
            traceback.print_exc()
    gc.collect()
    if sys.platform.startswith('linux'):    # hand freed heap pages back to the system (glibc keeps them)
        import ctypes
        try:
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except (OSError, AttributeError):
            pass


def run(project=None, engine=None, settings=None, variant=None, progress=None, cancel=None) -> ValidationResult:
    """Run the selected checks. project: the project whose robots, rig and exports the project-dependent checks look
    at (None: the built-in sample); the other checks build their own references. Returns the result (cancelled=True
    with the checks done so far when cancel() turns true)."""
    import datetime as _dt
    st = settings if isinstance(settings, ValidationSettings) else ValidationSettings.from_dict(settings)
    t0 = time.perf_counter()
    started = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
    ctx = Ctx(project, engine, variant, st)
    sel = specs(st)
    out, cancelled = [], False
    try:
        for k, spec in enumerate(sel):
            if cancel is not None and cancel():
                cancelled = True
                break
            if k and spec.model != sel[k - 1].model:
                release_memory(ctx)
            out.append(run_one(spec, ctx))
            if progress is not None:
                progress((k + 1) / max(len(sel), 1))
    finally:
        ctx.close()
    p = ctx.project
    return ValidationResult(out, st, getattr(p, 'name', '') if ctx.given else 'sample (built in)',
                            getattr(p.variant, 'name', '') if p is not None else '', started,
                            time.perf_counter() - t0, environment(), cancelled)


# ----------------------------------------------------------------------------- shared helpers for the checks
def series(label, x, y, kind='line'):
    """A figure series, rounded for the JSON (8 significant digits)."""
    return {'label': label, 'kind': kind, 'x': [float(f'{v:.8g}') for v in np.asarray(x, float).ravel()],
            'y': [float(f'{v:.8g}') if _finite(v) else None for v in np.asarray(y, float).ravel()]}


def figure(title, xlabel, ylabel, data, *, logy=False, hline=None, hline_label='', diag=False, kind='xy',
           categories=None):
    """A plot for the report: kind 'xy' (lines and points) or 'bars' (one value per category). hline: a reference
    line (the tolerance); diag: the y = x line (predicted against measured)."""
    return {'kind': kind, 'title': title, 'xlabel': xlabel, 'ylabel': ylabel, 'series': data, 'logy': logy,
            'hline': hline, 'hline_label': hline_label, 'diag': diag, 'categories': categories}


def rel_err(a, b) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-300)


def qt_app():
    """A QApplication when none runs (the cube maps and some exports paint with Qt); offscreen in a terminal."""
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
            os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        app = QApplication(sys.argv[:1])
    return app
