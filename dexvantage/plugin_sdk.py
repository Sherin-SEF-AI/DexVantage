"""Plugins: third-party packages that add studies, metrics, exporters, dataset readers, library folders and UI pages to
DexVantage without editing it.

A plugin is an installed Python distribution that declares an entry point in the group ``dexvantage.plugins``:

    # the plugin's pyproject.toml
    [project.entry-points."dexvantage.plugins"]
    my_plugin = "my_plugin:register"

    # my_plugin/__init__.py
    def register(api):                     # api: a PluginAPI, one per plugin
        api.study('my_study', 'My study', batch='my-study')(my_study_fn)
        api.metric('my.figure', 'My figure', 'mm', 'lower', study='my_study')(lambda doc: doc['summary']['mm'])
        api.exporter('my_format', 'My format')(my_export_fn)
        api.dataset_reader('my_format', sniff=my_sniff)(my_open_fn)
        api.library_folder(os.path.join(os.path.dirname(__file__), 'library'))
        api.page('my_page', 'My study', 'my_plugin.page:MyPage', menu='Analysis')   # the class is imported only
                                                                                     # when the window is built

The entry point may name a function or a module (its ``register`` is called). DexVantage loads plugins once at start-up
(app.py, before the batch flags are built, so a plugin study gets its ``--flag`` too), and the Python API loads them on
first use (dexvantage.api). Help ▸ Plugins lists every plugin found: its name, version, what it added and, for a plugin
that failed, why, in a plain sentence. A plugin that fails is rolled back (what it added before the failure is removed)
and the app carries on; its traceback goes to the log (stderr), never to the UI.

``--no-plugins`` (or the environment variable DEXVANTAGE_NO_PLUGINS=1) finds the installed plugins without loading
them: they are listed as "off".

Rules a plugin follows (checked where they can be): ids are new (a plugin cannot replace a built-in study, metric,
exporter or dataset format: the duplicate is skipped and said so); study ids are lower-case words joined by
underscores (no dots: factor paths and settings use the dot); a page class is given as "module:Class" so headless use
never imports Qt; a plugin's figures follow docs/DEVELOPING.md like DexVantage's own (cited or None). A plugin may set
``REQUIRES_API = n`` in its module; one that needs a newer plugin API than API_VERSION is refused with a sentence.

API
    load(enabled=True, entry_points=None, paths=None) -> [PluginRecord]   once; again with reload=True
    loaded() -> [PluginRecord]                         what the last load found (for Help ▸ Plugins and dv.plugins())
    discover(paths=None) -> [EntryPoint]               the installed entry points of the group (paths: search there)
    unload(name)                                       remove what a plugin added (tests; a plugin turned off)
    origin(kind, id) -> plugin name | None             which plugin added a study / metric / exporter / reader / page
    summary_text(record) -> str                        "2 studies (a, b), 1 metric (c)": what it added, in words

Known limits: a plugin runs in the same process with the same rights as DexVantage (there is no sandbox: install only
plugins you trust); unloading removes what a plugin registered, not modules it imported or state it changed in other
ways; UI pages are added when the window is built, so a plugin loaded later (from the Python API in a running window)
adds its page only after a restart; a page whose id equals a built-in page's id is replaced by the built-in one.
"""
from __future__ import annotations

import importlib
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field

GROUP = 'dexvantage.plugins'
API_VERSION = 1
ENV_OFF = 'DEXVANTAGE_NO_PLUGINS'
KINDS = ('study', 'metric', 'exporter', 'dataset_reader', 'library_folder', 'page')
KIND_WORDS = {'study': ('study', 'studies'), 'metric': ('metric', 'metrics'), 'exporter': ('exporter', 'exporters'),
              'dataset_reader': ('dataset reader', 'dataset readers'),
              'library_folder': ('library folder', 'library folders'), 'page': ('page', 'pages')}
STATUS_TEXT = {'loaded': 'loaded', 'error': 'could not be loaded', 'off': 'off'}
STATUS_LEVEL = {'loaded': 'good', 'error': 'critical', 'off': 'warning'}
_STUDY_ID = re.compile(r'^[a-z][a-z0-9_]*$')


class PluginError(ValueError):
    """A plugin asked for something DexVantage cannot give it; the message says what, in plain words."""


@dataclass
class PluginRecord:
    name: str                       # the entry point's name
    value: str = ''                 # 'module:function'
    distribution: str = ''          # the installed distribution's name, when known
    version: str = ''               # its version, else the module's __version__, else ''
    status: str = 'off'             # 'loaded' | 'error' | 'off'
    error: str = ''                 # why it could not be loaded, a plain sentence starting with its name
    reason: str = ''                # the same without the name ('KeyError: ...', 'its module could not be imported')
    problems: list = field(default_factory=list)      # what was skipped (a duplicate id), plain sentences
    added: dict = field(default_factory=lambda: {k: [] for k in KINDS})
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {'name': self.name, 'value': self.value, 'distribution': self.distribution, 'version': self.version,
                'status': self.status, 'status_text': STATUS_TEXT[self.status], 'error': self.error,
                'reason': self.reason,
                'problems': list(self.problems), 'added': {k: list(v) for k, v in self.added.items() if v},
                'summary': summary_text(self), 'seconds': round(self.seconds, 4)}


_RECORDS: list = []
_LOADED = False
_ENABLED = True
ORIGIN: dict = {}                   # (kind, id) -> plugin name


def _first_line(e: BaseException) -> str:
    return ((str(e) or type(e).__name__).strip().splitlines() or [''])[0][:240]


def _builtins_loaded():
    """Import DexVantage's own registries first, so a plugin's duplicate id is caught and built-in readers sniff
    before a plugin's."""
    from .core import exporters as EX
    from .core import jobs
    from .core import metrics as M
    from .core.datasets import load_readers
    jobs.load_studies()
    M.load_metrics()
    EX.load_exporters()
    load_readers()


class PluginAPI:
    """What a plugin's register(api) receives: one per plugin, recording what it adds (shown in Help ▸ Plugins and
    removed again when the plugin fails or is unloaded)."""

    api_version = API_VERSION

    def __init__(self, record: PluginRecord):
        self._rec = record
        from . import __version__
        self.dexvantage_version = __version__

    # ---------------------------------------------------------------- helpers a plugin may use
    @property
    def StudyError(self):
        from .core.jobs import StudyError
        return StudyError

    @property
    def jobs(self):
        from .core import jobs
        return jobs

    @property
    def dv(self):
        """The Python API (dexvantage.api): dv.sample, Rig, Result, ..."""
        from . import api
        return api

    def _skip(self, kind: str, ident: str, why: str):
        self._rec.problems.append(f'{self._rec.name} tried to add the {KIND_WORDS[kind][0]} "{ident}", {why}; it was '
                                  'not added.')

    def _note(self, kind: str, ident: str):
        self._rec.added[kind].append(ident)
        ORIGIN[(kind, ident)] = self._rec.name

    # ---------------------------------------------------------------- what a plugin can add
    def study(self, sid: str, title: str, quick: bool = False, batch: str = '', needs_robot: bool = False):
        """Decorator: register fn(project, ctx) -> dict as a study (core.jobs.study). batch: its batch flag without
        dashes ('' for none)."""
        from .core import jobs

        def deco(fn):
            if not _STUDY_ID.match(str(sid)):
                raise PluginError(f'the study id "{sid}" is not lower-case words joined by underscores (a dot or a '
                                  'capital is not allowed: factor paths use the dot)')
            if sid in jobs.STUDIES:
                self._skip('study', sid, 'which DexVantage already has')
                return fn
            taken = {i.batch for i in jobs.STUDIES.values() if i.batch}
            if batch and batch in taken:
                self._skip('study', sid, f'with the batch flag --{batch}, which another study already uses')
                return fn
            jobs.study(sid, title, quick=quick, batch=batch, needs_robot=needs_robot)(fn)
            self._note('study', sid)
            return fn
        return deco

    def metric(self, mid: str, title: str, unit: str, better: str, study: str, note: str = '', tags=()):
        """Decorator: register extract(doc) -> float | None as a metric (core.metrics.metric)."""
        from .core import metrics as M

        def deco(fn):
            if mid in M.METRICS:
                self._skip('metric', mid, 'which DexVantage already has')
                return fn
            M.metric(mid, title, unit, better, study, note, tags)(fn)
            self._note('metric', mid)
            return fn
        return deco

    def exporter(self, fid: str, title: str, needs_robot: bool = False, description: str = ''):
        """Decorator: register fn(project, out_dir, options, ctx) -> {'files', 'notes'} (core.exporters.exporter)."""
        from .core import exporters as EX

        def deco(fn):
            if fid in EX.EXPORTERS:
                self._skip('exporter', fid, 'which DexVantage already has')
                return fn
            EX.exporter(fid, title, needs_robot, description)(fn)
            self._note('exporter', fid)
            return fn
        return deco

    def dataset_reader(self, fmt: str, sniff, title: str = ''):
        """Decorator: register open(path) -> core.datasets.Dataset for the format fmt; sniff(path) -> bool says
        whether a path is in it. title: the format's name in words ('' = fmt)."""
        from .core.datasets import episode as EP

        def deco(fn):
            if any(r[0] == fmt for r in EP.READERS):
                self._skip('dataset_reader', fmt, 'a format DexVantage already reads')
                return fn
            EP.register_reader(fmt, sniff)(fn)
            if title:
                EP.FORMAT_TITLES[fmt] = title
            self._note('dataset_reader', fmt)
            return fn
        return deco

    def library_folder(self, path: str):
        """Read library entries (one JSON per device, the library schema) from this folder too; read-only, like the
        built-in library. Entries whose id is already taken are skipped (Library problems say so)."""
        from .core import library as LB
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            raise PluginError(f'the library folder {os.path.basename(path)} does not exist')
        LB.add_folder(path, f'the plugin {self._rec.name}')
        self._note('library_folder', path)

    def page(self, pid: str, title: str, cls, menu: str = 'Analysis', icon: str | None = None,
             action_id: str | None = None, shortcut: str | None = None, workspace: str | None = None,
             order: int = 200):
        """A UI page (ui/pages registry). cls: a StudyPage subclass or "module:Class" (imported when the window is
        built: headless use of the plugin never imports Qt)."""
        from .ui import pages as PG
        if pid in PG.PAGES:
            self._skip('page', pid, 'which DexVantage already has')
            return cls
        factory = _LazyPage(cls, self._rec.name) if isinstance(cls, str) else cls
        PG.register(pid, title, factory, menu=menu, icon=icon, action_id=action_id, shortcut=shortcut,
                    workspace=workspace, order=order)
        self._note('page', pid)
        return cls


class _LazyPage:
    """A page class named as "module:Class", imported when the window first builds the page."""

    def __init__(self, path: str, plugin: str):
        self.path = path
        self.plugin = plugin
        self.__name__ = path

    def resolve(self):
        mod, _, attr = self.path.partition(':')
        if not attr:
            raise PluginError(f'the page "{self.path}" of {self.plugin} is not given as "module:Class"')
        return getattr(importlib.import_module(mod), attr)

    def __call__(self, ctx):
        return self.resolve()(ctx)


# ====================================================================== discovery and loading
def discover(paths=None) -> list:
    """The entry points of the group: installed ones, or those of the distributions found on paths (folders, as
    importlib.metadata searches them: a pip --target folder, a site-packages)."""
    from importlib import metadata
    if paths is None:
        eps = metadata.entry_points()
        return list(eps.select(group=GROUP) if hasattr(eps, 'select') else eps.get(GROUP, []))
    out = []
    for dist in metadata.distributions(path=[str(p) for p in paths]):
        out += [ep for ep in dist.entry_points if ep.group == GROUP]
    return out


def _dist_of(ep):
    d = getattr(ep, 'dist', None)
    if d is None:
        return '', ''
    try:
        return str(d.metadata['Name'] or ''), str(d.version or '')
    except Exception:                           # noqa: BLE001 — a broken METADATA must not stop the others
        return '', ''


def unload(name: str):
    """Remove everything the named plugin added (studies, metrics, exporters, readers, library folders, pages)."""
    for rec in [r for r in _RECORDS if r.name == name]:
        _unload_record(rec)


def _unload_record(rec: PluginRecord):
    from .core import exporters as EX
    from .core import jobs
    from .core import metrics as M
    a = rec.added
    for sid in a['study']:
        jobs.STUDIES.pop(sid, None)
    for mid in a['metric']:
        M.METRICS.pop(mid, None)
    for fid in a['exporter']:
        EX.EXPORTERS.pop(fid, None)
    if a['dataset_reader']:
        from .core.datasets import episode as EP
        EP.READERS[:] = [r for r in EP.READERS if r[0] not in a['dataset_reader']]
        for f in a['dataset_reader']:
            EP.FORMAT_TITLES.pop(f, None)
    if a['library_folder']:
        from .core import library as LB
        for p in a['library_folder']:
            LB.remove_folder(p)
    if a['page']:
        from .ui import pages as PG
        for pid in a['page']:
            PG.PAGES.pop(pid, None)
    for k in KINDS:
        for i in a[k]:
            ORIGIN.pop((k, i), None)
        a[k] = []


def _load_one(ep, rec: PluginRecord):
    t0 = time.perf_counter()
    try:
        try:
            obj = ep.load()
        except ModuleNotFoundError as e:
            raise PluginError(f'its module could not be imported ({_first_line(e)}): reinstall it, or remove it with '
                              f'pip uninstall {rec.distribution or rec.name}') from e
        fn = getattr(obj, 'register', obj)
        if not callable(fn):
            raise PluginError(f'its entry point {rec.value} names neither a register(api) function nor a module '
                              'that has one')
        mod = sys.modules.get(getattr(fn, '__module__', '') or '')
        if not rec.version:
            rec.version = str(getattr(mod, '__version__', '') or '')
        need = getattr(mod, 'REQUIRES_API', getattr(obj, 'REQUIRES_API', 1))
        if isinstance(need, int) and need > API_VERSION:
            raise PluginError(f'it needs version {need} of the plugin interface and this DexVantage has version '
                              f'{API_VERSION}: update DexVantage')
        fn(PluginAPI(rec))
        rec.status = 'loaded'
    except Exception as e:                      # noqa: BLE001 — a plugin must never stop the app; explained, logged
        traceback.print_exc()
        unload_rec = [k for k in KINDS if rec.added[k]]
        _unload_record(rec)
        rec.status = 'error'
        why = str(e) if isinstance(e, PluginError) else f'{type(e).__name__}: {_first_line(e)}'
        rec.reason = why.rstrip('. ')
        rec.error = f'{rec.name} could not be loaded: {rec.reason}.'
        if unload_rec:
            rec.error += ' What it had added before the failure was removed.'
    rec.seconds = time.perf_counter() - t0


def load(enabled: bool | None = None, entry_points=None, paths=None, reload: bool = False) -> list:
    """Find and load the plugins once (again with reload=True, which unloads the previous ones first). enabled=None
    follows DEXVANTAGE_NO_PLUGINS; False finds them without loading (status 'off'). entry_points: load these
    instead of the installed ones (tests); paths: find them on these folders instead of sys.path."""
    global _LOADED, _ENABLED
    if _LOADED and not reload and entry_points is None and paths is None:
        return list(_RECORDS)
    if enabled is None:
        enabled = os.environ.get(ENV_OFF, '').strip().lower() not in ('1', 'true', 'yes', 'on')
    for r in list(_RECORDS):
        unload(r.name)
    _RECORDS.clear()
    _ENABLED = bool(enabled)
    try:
        eps = list(entry_points) if entry_points is not None else discover(paths)
    except Exception as e:                      # noqa: BLE001 — a broken installation must not stop the app
        traceback.print_exc()
        why = f'{type(e).__name__}: {_first_line(e)}'
        rec = PluginRecord('plugins', status='error', reason=f'the installed plugins could not be listed ({why})',
                           error=f'The installed plugins could not be listed: {why}.')
        _RECORDS.append(rec)
        _LOADED = True
        return list(_RECORDS)
    if enabled and eps:
        _builtins_loaded()
    seen: set = set()
    for ep in eps:
        dist, ver = _dist_of(ep)
        rec = PluginRecord(name=ep.name, value=getattr(ep, 'value', ''), distribution=dist, version=ver)
        if ep.name in seen:
            rec.status = 'error'
            rec.reason = (f'another installed plugin already uses the name {ep.name} (only the first is loaded)'
                          + (f'; this one is from {dist}' if dist else ''))
            rec.error = f'{ep.name} could not be loaded: {rec.reason}.'
        elif not enabled:
            rec.status = 'off'
        else:
            _load_one(ep, rec)
        seen.add(ep.name)
        _RECORDS.append(rec)
    _LOADED = True
    return list(_RECORDS)


def ensure_loaded() -> list:
    """load() unless it ran already (the Python API calls this on first use)."""
    return load() if not _LOADED else list(_RECORDS)


def loaded() -> list:
    return list(_RECORDS)


def enabled() -> bool:
    """False when this session was started with --no-plugins (or DEXVANTAGE_NO_PLUGINS)."""
    return _ENABLED


def origin(kind: str, ident: str) -> str | None:
    return ORIGIN.get((kind, ident))


def summary_text(rec: PluginRecord) -> str:
    """What the plugin added, in words: '1 study (aim_check), 1 metric (aim.worst_deg), 1 page (aim_check)'."""
    parts = []
    for k in KINDS:
        ids = rec.added.get(k) or []
        if not ids:
            continue
        one, many = KIND_WORDS[k]
        shown = [os.path.basename(i) if k == 'library_folder' else i for i in ids]
        parts.append(f'{len(ids)} {one if len(ids) == 1 else many} ({", ".join(shown)})')
    if parts:
        return ', '.join(parts)
    return {'loaded': 'nothing', 'error': 'nothing (it failed)', 'off': 'nothing (plugins are off)'}[rec.status]


def report_lines() -> list:
    """One plain line per plugin, for the terminal (--list-plugins) and the self-test."""
    out = []
    for r in _RECORDS:
        ver = f' {r.version}' if r.version else ''
        line = f'{r.name}{ver}: {STATUS_TEXT[r.status]}'
        if r.status == 'loaded':
            line += f'; added {summary_text(r)}'
        if r.status == 'error':
            line += f': {r.reason or r.error}.'
            if r.error.endswith('was removed.'):
                line += ' What it had added before the failure was removed.'
        for p in r.problems:
            line += f' {p}'
        out.append(line)
    return out
