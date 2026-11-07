"""The Python and notebook API: DexVantage without the window (no Qt is imported on this path).

    import dexvantage as dv

    rig = dv.sample('ur5e')                          # or dv.open('cell.dexv'), dv.new('My cell')
    res = rig.run('coverage')                        # a Result: a dict (the study's JSON) plus .headline, .to_frame()
    res.headline['text']                             # 'Task volume seen: 91.9 % (mostly single or blind)'
    rig.metrics(['coverage.covered', 'bandwidth.payload_mbps'])
    rig.factors['sensor:Overhead D435.z']            # read a factor (core/factors.py paths)
    taller = rig.with_factors({'sensor:Overhead D435.z': 1.2})   # a NEW Rig; rig is unchanged
    rig.factors['sensor:Overhead D435.z'] = 1.2      # edits rig itself (its project is replaced by the edited one)
    rig.export('out/', ['opencv_yaml', 'mjcf'])      # core/exporters: files, notes, a manifest with SHA-256
    rig.check('requirements.yaml')                   # the readiness scorecard (core/requirements.py)
    rig.save('cell.dexv')
    dv.studies(), dv.metrics(), dv.exporters(), dv.plugins()
    ds = dv.datasets.open('path/to/lerobot_dataset')  # LeRobot v2.1 / v3.0, RLDS, ROS 2 bags, plugin formats

What edits in place and what returns a new object:

- ``rig.factors[path] = value`` and ``rig.factors.update({...})`` change ``rig`` (its project is replaced by the
  edited copy core.factors.apply returns; Robot and Sensor objects read from it before are then stale).
- ``rig.with_factors({...})``, ``rig.use_variant(name)`` and ``rig.copy()`` return a new Rig; ``rig`` is unchanged.
- ``rig.run(study, **settings)`` runs on a copy with the settings laid over the stored ones; the rig's stored
  settings are unchanged (store them with ``rig.factors['setting:<study>.<key>'] = value``).
- ``rig.robots``, ``rig.sensors`` and ``rig.variants`` are the project's own objects: changing one changes the rig.

Every number comes from the studies themselves (the same code the window and batch mode run, through core.jobs and its
result cache, keyed by the project's content): the API computes nothing of its own. A study that cannot run raises
DexVantageError (core.jobs.StudyError) with a plain sentence; a figure a result does not state reads None.

Rig and Result have rich HTML views for Jupyter (_repr_html_). Result.to_frame() returns a pandas DataFrame when
pandas is installed (the optional extra "research": pip install "dexvantage[research]"), else a list of dicts (Records)
whose .note says so.

Doctest (run by tests/test_api.py):

>>> import dexvantage as dv
>>> rig = dv.sample('so100')
>>> rig
<Rig 'SO-100 sample': 1 robot, 2 sensors, 1 variant>
>>> [s.name for s in rig.sensors]
['Wrist camera (sample placement)', 'Third-person camera (sample placement)']
>>> z = 'sensor:Third-person camera (sample placement).z'
>>> rig.factors[z]
0.36
>>> higher = rig.with_factors({z: 0.5})               # a new Rig
>>> higher.factors[z], rig.factors[z]
(0.5, 0.36)
>>> rig.factors[z] = 0.4                               # edits rig itself
>>> rig.factors[z]
0.4
>>> res = rig.run('bandwidth')
>>> isinstance(res, dict), res['study'], res.headline['word']
(True, 'bandwidth', 'no hosts')
>>> res.headline['text']
'Smallest controller margin: not stated (no hosts). The rig has no host controller: ...'
>>> 'coverage' in [s['id'] for s in dv.studies()]
True

Known limits: the API is a thin layer over the studies, so their own Known limits apply; studies run in the calling
thread (the window's background controller is not used: a long study blocks the notebook cell, pass progress= to see
it move); the result cache is per process (32 results, keyed by content, so a changed project or setting misses it);
UI pages a plugin adds are not reachable from here; HTML views show at most 200 rows of a table.
"""
from __future__ import annotations

import builtins
import html
import json
import math
import os

from .core import jobs
from .core.jobs import StudyError

DexVantageError = StudyError
__all__ = ['Rig', 'Result', 'Records', 'Table', 'MetricTable', 'Recording', 'DexVantageError', 'open', 'sample',
           'new', 'samples', 'studies', 'metrics', 'exporters', 'plugins', 'load_plugins', 'datasets']

_CACHE = jobs.ResultCache(32)
_HTML_ROWS = 200
STATUS_HEX = {'good': '#0ca30c', 'warning': '#fab219', 'serious': '#ec835a', 'critical': '#d03b3b'}   # ui/theme
PANDAS_NOTE = ('pandas is not installed, so this is a list of dicts (one per row): install the research extra, '
               'pip install "dexvantage[research]", for a DataFrame.')


def _plugins():
    from . import plugin_sdk as PL
    PL.ensure_loaded()


# ====================================================================== tables
class Records(list):
    """A list of row dicts, returned by to_frame() when pandas is not installed; .note says so."""
    note = PANDAS_NOTE


def _flatten(d: dict, prefix: str = '') -> dict:
    out = {}
    for k, v in d.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            out.update(_flatten(v, key + '.'))
        else:
            out[key] = v
    return out


def _frame(rows: list, columns=None):
    """A pandas DataFrame of the row dicts when pandas is installed, else Records."""
    rows = [_flatten(r) if isinstance(r, dict) else {'value': r} for r in rows]
    if columns is not None:
        rows = [{c: r.get(c) for c in columns} for r in rows]
    try:
        import pandas as pd
    except ImportError:
        return Records(rows)
    return pd.DataFrame(rows, columns=list(columns) if columns is not None else None)


def _cell(v) -> str:
    if v is None:
        return '<span title="not stated">—</span>'
    if isinstance(v, float):
        if not math.isfinite(v):
            return '—'
        return html.escape(f'{v:.6g}')
    if isinstance(v, (list, tuple)):
        s = json.dumps(v, default=str)
        return html.escape(s if len(s) <= 80 else s[:77] + '…')
    if isinstance(v, dict):
        return html.escape(f'{{{len(v)} keys}}')
    return html.escape(str(v))


def _html_table(rows: list, columns=None) -> str:
    rows = [_flatten(r) if isinstance(r, dict) else {'value': r} for r in rows]
    if columns is None:
        columns = []
        for r in rows:
            columns += [k for k in r if k not in columns]
    head = ''.join(f'<th style="text-align:left;padding:2px 8px">{html.escape(str(c))}</th>' for c in columns)
    body = ''.join('<tr>' + ''.join(f'<td style="padding:2px 8px;font-family:monospace">{_cell(r.get(c))}</td>'
                                    for c in columns) + '</tr>' for r in rows[:_HTML_ROWS])
    more = (f'<p style="opacity:.7">{len(rows) - _HTML_ROWS} more rows not shown (to_frame() has them all).</p>'
            if len(rows) > _HTML_ROWS else '')
    return f'<table style="border-collapse:collapse"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{more}'


def _status_html(level: str, word: str) -> str:
    c = STATUS_HEX.get(level or '', '#888888')
    return (f'<span style="border:1px solid {c};color:{c};border-radius:3px;padding:0 5px;font-size:90%">'
            f'{html.escape(word or level or "")}</span>')


class Table(list):
    """A list of row dicts (dv.studies(), dv.metrics(), ...) with to_frame() and an HTML view."""

    def to_frame(self):
        return _frame(list(self))

    def _repr_html_(self) -> str:
        return _html_table(list(self))


class MetricTable(dict):
    """{metric id: {'value', 'unit', 'better', 'study', 'title', 'status', 'reason'?, 'bound'?}} (core.metrics), with
    to_frame() (one row per metric) and an HTML view."""

    def rows(self) -> list:
        return [{'id': k, **{c: v.get(c) for c in ('title', 'value', 'unit', 'better', 'status', 'bound', 'study',
                                                     'reason')}} for k, v in self.items()]

    def figures(self) -> dict:
        """{metric id: value} (None where not stated)."""
        return {k: v.get('value') for k, v in self.items()}

    def to_frame(self):
        return _frame(self.rows())

    def _repr_html_(self) -> str:
        return _html_table(self.rows(), ['id', 'title', 'value', 'unit', 'better', 'status', 'reason'])


# ====================================================================== results
def _status_of(doc: dict) -> tuple:
    """(level, word) a result states for itself: its headline's or summary's status and word, else ('', '')."""
    level = word = ''
    for part in (doc.get('headline'), doc.get('summary'), doc):
        if not isinstance(part, dict):
            continue
        for k in ('status_word', 'word', 'verdict_text'):
            if not word and isinstance(part.get(k), str):
                word = part[k]
        for k in ('status', 'level'):
            if not level and part.get(k) in STATUS_HEX:
                level = part[k]
    return level, word


def _fmt(v, unit: str) -> str:
    if v is None:
        return 'not stated'
    if float(v).is_integer() and abs(v) < 1e9:
        t = f'{int(v)}'
    elif abs(v) < 10:
        t = f'{v:.3g}'
    else:
        t = f'{v:.1f}'
    if unit == '°':
        return f'{t}°'
    return f'{t} {unit}' if unit and unit != 'count' else t


class Result(dict):
    """A study's result: the study's own JSON document (a dict: res['summary'], json.dumps(res)) plus

        .study, .title        the study id and its title
        .headline             {'metric', 'title', 'value', 'unit', 'better', 'status', 'bound', 'level', 'word',
                               'text'}: the study's headline figure (its metric tagged 'headline') with the status
                               word the result gives, or None when the study has no headline metric
        .metrics()            every metric of this study read off this result (no re-run): a MetricTable
        .to_frame(key=None, columns=None)
                              key None: the metrics, one row each; else the list (or dict) under that key, one row
                              per item, nested fields as dotted columns (columns= keeps only those): a pandas
                              DataFrame, or Records
        .tables()             the keys that hold lists of rows (what to_frame(key) takes)
        .save(path)           JSON with the provenance block (core/provenance.py)
    """

    def __init__(self, doc: dict, study: str = '', title: str = '', project=None):
        super().__init__(doc)
        self.study = study or str(doc.get('study') or '')
        info = jobs.STUDIES.get(self.study)
        self.title = title or (info.title if info else self.study)
        self._project = project

    # ---------------------------------------------------------------- figures
    def metrics(self) -> MetricTable:
        from .core import metrics as M
        M.load_metrics()
        out = MetricTable()
        for m in M.METRICS.values():
            if m.study == self.study:
                out[m.id] = {'value': None, 'unit': m.unit, 'better': m.better, 'study': m.study, 'title': m.title,
                             **M.read(m, dict(self))}
        return out

    @property
    def headline(self) -> dict | None:
        from .core import metrics as M
        M.load_metrics()
        level, word = _status_of(self)
        m = next((m for m in M.METRICS.values() if m.study == self.study and 'headline' in m.tags), None)
        if m is None:                                  # no headline metric: the result's own headline, if any
            if not word:
                return None
            own = self.get('headline') if isinstance(self.get('headline'), dict) else {}
            shown = own.get('value')
            text = f'{self.title}: ' + (f'{shown} ({word})' if shown not in (None, '') else word)
            if own.get('text'):
                text += f'. {own["text"]}'
            return {'metric': None, 'title': self.title, 'value': shown, 'unit': '', 'better': None,
                    'status': 'ok' if shown not in (None, '') else 'not stated', 'bound': None, 'level': level,
                    'word': word, 'text': text}
        r = M.read(m, dict(self))
        v = r.get('value')
        bound = r.get('bound')
        shown = _fmt(v, m.unit)
        if bound and v is not None:
            shown = ('≥ ' if bound == 'at_least' else '≤ ') + shown
        text = f'{m.title}: {shown}' + (f' ({word})' if word else '')
        if v is None and r.get('reason'):
            text += f'. {r["reason"]}'
        return {'metric': m.id, 'title': m.title, 'value': v, 'unit': m.unit, 'better': m.better,
                'status': r['status'], 'bound': bound, 'level': level, 'word': word, 'text': text}

    # ---------------------------------------------------------------- tables
    def tables(self) -> list:
        return [k for k, v in self.items() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)]

    def rows(self, key: str | None = None) -> list:
        if key is None:
            return self.metrics().rows()
        if key not in self:
            raise DexVantageError(f'The {self.title.lower()} result has no "{key}" (its tables are: '
                                  f'{", ".join(self.tables()) or "none"}).')
        v = self[key]
        if isinstance(v, list):
            return list(v)
        if isinstance(v, dict):
            if v and all(isinstance(x, dict) for x in v.values()):
                return [{'key': k, **x} for k, x in v.items()]
            return [v]
        return [{key: v}]

    def to_frame(self, key: str | None = None, columns=None):
        """A table of the result (the class docstring); columns: keep only these (dotted names for nested fields)."""
        return _frame(self.rows(key), columns)

    def save(self, path: str) -> str:
        """Write the result as JSON with the provenance block; returns the path."""
        from .core import provenance as PV
        doc = {k: v for k, v in self.items() if k != 'cached'}
        if self._project is not None:
            doc = PV.stamp(doc, self._project, output={'option': None, 'name': os.path.basename(path),
                                                       'in_folder': False})
        with builtins.open(path, 'w', encoding='utf-8') as fh:       # this module's open() is dv.open
            json.dump(doc, fh, indent=1, ensure_ascii=False, default=str)
            fh.write('\n')
        return path

    # ---------------------------------------------------------------- views
    def __repr__(self) -> str:
        h = self.headline
        return f'<Result {self.study}: {h["text"] if h else "no headline figure"}>'

    def _repr_html_(self) -> str:
        h = self.headline
        out = [f'<div><div style="opacity:.7">{html.escape(self.title)}</div>']
        if h:
            fig = _cell(h['value']) if h['value'] is not None else '—'
            if h.get('bound') and h['value'] is not None:
                fig = ('≥ ' if h['bound'] == 'at_least' else '≤ ') + fig
            out.append(f'<div style="font-size:160%;font-family:monospace">{fig} '
                       f'<span style="font-size:60%">{html.escape(h["unit"] or "")}</span> '
                       f'{_status_html(h["level"], h["word"] or h["status"])}</div>'
                       f'<div>{html.escape(h["text"])}</div>')
        ms = self.metrics()
        if ms:
            out.append(ms._repr_html_())
        sec = self.get('seconds')
        tail = [f'{sec:.2f} s' if isinstance(sec, (int, float)) else '', 'from the cache' if self.get('cached') else '',
                f'tables: {", ".join(self.tables())}' if self.tables() else '']
        out.append(f'<div style="opacity:.7">{html.escape(" · ".join(t for t in tail if t))}</div></div>')
        return ''.join(out)


# ====================================================================== factors
class Factors:
    """The rig's settable quantities (core/factors.py paths): rig.factors[path] reads; rig.factors[path] = value and
    update({...}) edit the rig in place; apply({...}) returns a new Rig."""

    def __init__(self, rig: 'Rig'):
        self._rig = rig

    def __getitem__(self, path: str):
        from .core import factors as F
        return F.get(self._rig.project, path)

    def __setitem__(self, path: str, value):
        self.update({path: value})

    def update(self, values: dict, allow_locked: bool = False):
        """Set every factor on the rig itself (in the order given)."""
        from .core import factors as F
        self._rig._p = F.apply(self._rig.project, dict(values), allow_locked=allow_locked)

    def apply(self, values: dict, allow_locked: bool = False) -> 'Rig':
        """A new Rig with the factors set; this rig is unchanged."""
        from .core import factors as F
        return Rig(F.apply(self._rig.project, dict(values), allow_locked=allow_locked))

    def describe(self, path: str) -> dict:
        from .core import factors as F
        return F.describe(path, self._rig.project)

    def paths(self) -> Table:
        """Every sensor factor of the active variant and every stored study setting, with its current value."""
        from .core import factors as F
        p = self._rig.project
        rows = []
        names = [s.name for s in p.variant.sensors]
        for s in p.variant.sensors:
            ident = s.name if names.count(s.name) == 1 else s.uid
            for fld in F.POSITION + F.ANGLES + ('model', 'parent', 'enabled', 'stream'):
                path = f'sensor:{ident}.{fld}'
                d = F.describe(path, p)
                rows.append({'path': path, 'title': d['title'], 'value': F.get(p, path), 'unit': d['unit'],
                             'kind': d['kind'], 'locked': d.get('locked', False)})
        for sid, st in sorted(p.analysis.studies.items()):
            for k, v in sorted((st or {}).items()):
                rows.append({'path': f'setting:{sid}.{k}', 'title': f'{sid} setting {k}', 'value': v, 'unit': '',
                             'kind': 'setting', 'locked': False})
        return Table(rows)

    def __iter__(self):
        return iter(r['path'] for r in self.paths())

    def __repr__(self) -> str:
        return f'<Factors of {self._rig.name!r}: {len(self.paths())} paths (factors.paths() lists them)>'


# ====================================================================== the rig
def _study_id(name: str) -> str:
    jobs.load_studies()
    _plugins()
    key = str(name or '').strip()
    if key in jobs.STUDIES:
        return key
    alt = key.replace('-', '_')
    if alt in jobs.STUDIES:
        return alt
    for info in jobs.STUDIES.values():
        if info.batch and info.batch == key.replace('_', '-'):
            return info.id
    raise DexVantageError(f'There is no study called "{name}". The studies are: {", ".join(sorted(jobs.STUDIES))} '
                          '(dv.studies() lists them with their titles).')


class Rig:
    """A DexVantage project (core.project.Project) seen from Python: its robots, sensors and variants, its factors,
    its studies, metrics, exports and requirements. Rig(project) wraps a Project; dv.open / dv.sample / dv.new make
    one."""

    def __init__(self, project):
        from .core.project import Project
        if not isinstance(project, Project):
            raise DexVantageError('Rig() takes a DexVantage Project: use dv.open(path), dv.sample(name) or '
                                  'dv.new(name).')
        self._p = project

    # ---------------------------------------------------------------- making one
    @classmethod
    def open(cls, path: str) -> 'Rig':
        from .core.project import Project, ProjectFormatError
        try:
            return cls(Project.load(os.fspath(path)))
        except ProjectFormatError as e:                 # one exception type for a script to catch
            raise DexVantageError(str(e)) from None

    @classmethod
    def sample(cls, name: str) -> 'Rig':
        from .core.project import ProjectFormatError
        from .core.robot.samples import sample_project
        try:
            return cls(sample_project(name))
        except ProjectFormatError as e:                 # SampleError is one
            raise DexVantageError(str(e)) from None

    @classmethod
    def new(cls, name: str = 'Untitled') -> 'Rig':
        from .core.project import blank_project
        return cls(blank_project(name))

    def copy(self) -> 'Rig':
        return Rig(self._p.snapshot())

    # ---------------------------------------------------------------- what it holds
    @property
    def project(self):
        """The underlying core.project.Project (the same object: changing it changes the rig)."""
        return self._p

    @property
    def name(self) -> str:
        return self._p.name

    @property
    def path(self) -> str:
        return self._p.path

    @property
    def robots(self) -> list:
        return list(self._p.robots)

    @property
    def sensors(self) -> list:
        """The active variant's sensors."""
        return list(self._p.variant.sensors)

    @property
    def variants(self) -> list:
        return list(self._p.variants)

    @property
    def variant(self):
        return self._p.variant

    def use_variant(self, ident: str) -> 'Rig':
        """A new Rig with the variant named (by name or uid) active."""
        v = self._p.variant_by_uid(ident) or next((x for x in self._p.variants if x.name == ident), None)
        if v is None:
            raise DexVantageError(f'The rig has no variant "{ident}" (it has: '
                                  f'{", ".join(x.name for x in self._p.variants)}).')
        p = self._p.snapshot()
        p.active_uid = v.uid
        return Rig(p)

    @property
    def factors(self) -> Factors:
        return Factors(self)

    def with_factors(self, values: dict, allow_locked: bool = False) -> 'Rig':
        """A new Rig with these factors set (core/factors.py paths); this rig is unchanged."""
        return self.factors.apply(values, allow_locked=allow_locked)

    def sensor_rows(self) -> list:
        from .core.frames import rpy_from_quat
        rows = []
        for s in self._p.variant.sensors:
            p = s.mount.pose
            rpy = [math.degrees(a) for a in rpy_from_quat(p.quaternion)]
            rows.append({'name': s.name, 'uid': s.uid, 'kind': s.kind, 'model': s.model_id or None,
                         'parent': s.mount.parent_frame, 'x_m': float(p.position[0]), 'y_m': float(p.position[1]),
                         'z_m': float(p.position[2]), 'roll_deg': rpy[0], 'pitch_deg': rpy[1], 'yaw_deg': rpy[2],
                         'enabled': s.enabled, 'locked': s.locked})
        return rows

    def to_frame(self):
        """The active variant's sensors, one row each (mount pose in the parent frame)."""
        return _frame(self.sensor_rows())

    # ---------------------------------------------------------------- studies and figures
    def run(self, study: str, *, progress=None, **settings) -> Result:
        """Run a study (its id, e.g. 'coverage', or its batch flag, e.g. 'grasp') on a copy of the rig with
        settings laid over the stored ones; the rig is unchanged. progress: fn(fraction, text)."""
        sid = _study_id(study)
        p = self._p.snapshot()
        if settings:
            st = dict(p.analysis.studies.get(sid) or {})
            st.update(settings)
            p.analysis.studies[sid] = st
        doc = jobs.run_study(sid, p, ctx=jobs.Ctx(progress=progress), cache=_CACHE)
        return Result(doc, sid, project=p)

    def metrics(self, ids) -> MetricTable:
        """The named metrics (core/metrics.py ids; dv.metrics() lists them), each study run at most once."""
        from .core import metrics as M
        _plugins()
        if isinstance(ids, str):
            ids = [ids]
        return MetricTable(M.evaluate(self._p.snapshot(), list(ids), cache=_CACHE))

    def export(self, out_dir: str, formats, options: dict | None = None) -> dict:
        """Write the rig in the given formats (dv.exporters() lists them) into out_dir; returns the manifest
        (formats with their files, notes and status; every file's SHA-256; export_manifest.json beside them)."""
        from .core import exporters as EX
        _plugins()
        if isinstance(formats, str):
            formats = [f.strip() for f in formats.split(',') if f.strip()]
        return EX.export(self._p.snapshot(), os.fspath(out_dir), list(formats), options=options)

    def check(self, requirements=None, variant: str | None = None) -> Result:
        """The readiness scorecard: requirements is a requirements file (YAML or JSON), a list of requirement dicts,
        'defaults' (DexVantage's labelled defaults) or None (the rig's own). Returns a Result whose headline is the
        verdict ('ready', 'not ready', 'not evaluated')."""
        try:
            from .core import requirements as RQ
        except ImportError:                                  # pragma: no cover - a build without requirements
            raise DexVantageError('This build has no requirements scorecard.') from None
        _plugins()
        if isinstance(requirements, (str, os.PathLike)):
            reqs = RQ.default_requirements() if str(requirements) == 'defaults' else RQ.load(os.fspath(requirements))
        else:
            reqs = None if requirements is None else list(requirements)
        card = RQ.check(self._p.snapshot(), reqs, cache=_CACHE, variant=variant)
        return Result({'schema': RQ.SCHEMA, 'study': 'requirements', **card}, 'requirements',
                      'Readiness scorecard', project=self._p)

    def synth_episodes(self, folder: str, progress=None, **settings) -> dict:
        """Render synthetic LeRobot episodes with known faults into a new or empty folder (core/sim/episodes.py;
        needs MuJoCo's offscreen renderer); returns its summary."""
        from .core.sim import episodes as E
        return E.generate(self._p.snapshot(), os.fspath(folder), settings or None, progress=progress)

    def save(self, path: str | None = None) -> str:
        """Write the .dexv file (to path, or where it was opened from; a folder or a path ending in .dexv.d: the
        unpacked, Git-friendly form); returns the path."""
        from .core.project import EXTENSION
        path = os.fspath(path) if path is not None else self._p.path
        if not path:
            raise DexVantageError('This rig has never been saved: give a path, rig.save("cell.dexv").')
        if not os.path.isdir(path) and not os.path.splitext(path)[1]:     # a folder: the unpacked form
            path += EXTENSION
        try:
            self._p.save(path)
        except (OSError, ValueError) as e:              # ProjectFormatError: a folder that is not a project's
            raise DexVantageError(f'Could not write {os.path.basename(path.rstrip(os.sep))}: '
                                  f'{getattr(e, "strerror", None) or e}') from None
        return path

    # ---------------------------------------------------------------- views
    def __repr__(self) -> str:
        n_r, n_s, n_v = len(self._p.robots), len(self._p.variant.sensors), len(self._p.variants)
        return (f'<Rig {self.name!r}: {n_r} robot{"s" * (n_r != 1)}, {n_s} sensor{"s" * (n_s != 1)}, '
                f'{n_v} variant{"s" * (n_v != 1)}>')

    def _repr_html_(self) -> str:
        p = self._p
        mov = jobs.MOVING_JOINTS
        robots = ', '.join(f'{html.escape(r.name)} ({sum(j.get("type") in mov for j in r.joints)} joints)'
                           for r in p.robots) or 'none'
        variants = ', '.join(('<b>' + html.escape(v.name) + '</b>') if v.uid == p.active_uid else html.escape(v.name)
                             for v in p.variants)
        vols = ', '.join(html.escape(v.name) for v in p.workspace.volumes) or 'none'
        return (f'<div><div style="font-size:130%">{html.escape(self.name)}</div>'
                f'<div>Robots: {robots}</div><div>Variants: {variants} (active in bold)</div>'
                f'<div>Task volumes: {vols}</div>'
                f'{_html_table(self.sensor_rows(), ["name", "kind", "model", "parent", "x_m", "y_m", "z_m", "roll_deg", "pitch_deg", "yaw_deg", "enabled", "locked"])}'  # noqa: E501
                '</div>')


# ====================================================================== datasets
class Recording:
    """A recorded dataset (core.datasets.Dataset) seen from Python: .format, .version, .path, .info, episodes()
    (a Table), episode(i) (a core.datasets.Episode), truth(), to_frame() (the episodes), close(). Use it in a with
    block, or close() it."""

    def __init__(self, ds):
        self.raw = ds

    format = property(lambda self: self.raw.format)
    version = property(lambda self: self.raw.version)
    path = property(lambda self: self.raw.path)
    info = property(lambda self: self.raw.info)

    def episodes(self) -> Table:
        return Table([{'index': e.index, 'id': e.id, 'length': e.length} for e in self.raw.episodes()])

    def episode(self, index: int):
        return self.raw.load(index)

    def truth(self):
        return self.raw.truth()

    def episode_rows(self, index: int) -> list:
        """One episode's joint-state rows: {'t', <joint name>: value, ...} per row."""
        ep = self.raw.load(index)
        names = list(ep.joint_names) or [f'q{i}' for i in range(ep.q.shape[1] if ep.q.ndim == 2 else 0)]
        return [{'t': float(t), **{n: float(v) for n, v in zip(names, row)}} for t, row in zip(ep.t, ep.q)]

    def to_frame(self, episode: int | None = None):
        """The episodes (one row each), or, with episode=i, that episode's joint-state rows."""
        return _frame(list(self.episodes()) if episode is None else self.episode_rows(episode))

    def close(self):
        self.raw.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self) -> str:
        from .core.datasets.episode import format_title
        return f'<Recording {format_title(self.format, self.version)} {os.path.basename(self.path or "")!r}: ' \
               f'{len(self.raw.episodes())} episodes>'

    def _repr_html_(self) -> str:
        from .core.datasets.episode import format_title
        return (f'<div><div style="font-size:130%">{html.escape(os.path.basename(self.path or ""))}</div>'
                f'<div>{html.escape(format_title(self.format, self.version))}</div>'
                f'{_html_table(list(self.episodes()))}</div>')


class _Datasets:
    """dv.datasets: open(path) -> Recording; formats() -> the formats DexVantage reads (plugins' too)."""

    def open(self, path: str) -> Recording:
        from .core.datasets import open_dataset
        _plugins()
        return Recording(open_dataset(os.fspath(path)))

    def formats(self) -> Table:
        from . import plugin_sdk as PL
        from .core.datasets import episode as EP
        from .core.datasets import load_readers
        load_readers()
        _plugins()
        return Table([{'format': f, 'title': EP.FORMAT_TITLES.get(f, f), 'plugin': PL.origin('dataset_reader', f)}
                      for f, _s, _o in EP.READERS])

    def __repr__(self) -> str:
        return '<dv.datasets: open(path), formats()>'


datasets = _Datasets()


# ====================================================================== module functions
def open(path: str) -> Rig:                                   # noqa: A001 — dv.open, as the API names it
    """Open a .dexv project as a Rig."""
    return Rig.open(path)


def sample(name: str) -> Rig:
    """A sample project as a Rig ('so100', 'ur5e', 'dexhand', 'aloha', 'panda-bimanual', 'g1'; dv.samples() lists
    them)."""
    return Rig.sample(name)


def new(name: str = 'Untitled') -> Rig:
    """An empty project as a Rig."""
    return Rig.new(name)


def samples() -> Table:
    from .core.robot.samples import list_samples
    return Table(list_samples())


def studies() -> Table:
    """Every study: id, title, batch flag, whether it needs a robot, and the plugin that added it (None: built in)."""
    from . import plugin_sdk as PL
    jobs.load_studies()
    _plugins()
    return Table([{'id': i.id, 'title': i.title, 'batch': f'--{i.batch}' if i.batch else None,
                   'needs_robot': i.needs_robot, 'quick': i.quick, 'plugin': PL.origin('study', i.id)}
                  for i in sorted(jobs.STUDIES.values(), key=lambda i: i.id)])


def metrics() -> Table:
    """Every metric: id, title, unit, better, study, note and the plugin that added it."""
    from . import plugin_sdk as PL
    from .core import metrics as M
    M.load_metrics()
    _plugins()
    return Table([{'id': m.id, 'title': m.title, 'unit': m.unit, 'better': m.better, 'study': m.study,
                   'headline': 'headline' in m.tags, 'note': m.note, 'plugin': PL.origin('metric', m.id)}
                  for m in sorted(M.METRICS.values(), key=lambda m: m.id)])


def exporters() -> Table:
    from . import plugin_sdk as PL
    from .core import exporters as EX
    EX.load_exporters()
    _plugins()
    return Table([{'id': e.id, 'title': e.title, 'needs_robot': e.needs_robot, 'plugin': PL.origin('exporter', e.id)}
                  for e in sorted(EX.EXPORTERS.values(), key=lambda e: e.id)])


def plugins() -> Table:
    """The plugins found at start-up (or on first use of the API): name, version, status, what they added, errors."""
    from . import plugin_sdk as PL
    PL.ensure_loaded()
    return Table([r.to_dict() for r in PL.loaded()])


def load_plugins(enabled: bool = True, reload: bool = True) -> Table:
    """Find and load the installed plugins again (enabled=False: list them without loading)."""
    from . import plugin_sdk as PL
    PL.load(enabled=enabled, reload=reload)
    return plugins()
