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
    dv.diff(rig, taller), dv.merge(base, ours, theirs)   # project diff and three-way merge (api_research.py)
    dv.bibtex(rig), dv.figure(res, 'fig.svg')        # BibTeX of every source; a vector figure of a result

What edits in place and what returns a new object:

- ``rig.factors[path] = value`` and ``rig.factors.update({...})`` change ``rig`` (its project is replaced by the
  edited copy core.factors.apply returns; Robot and Sensor objects read from it before are then stale).
- ``rig.with_factors({...})``, ``rig.use_variant(name)`` and ``rig.copy()`` return a new Rig; ``rig`` is unchanged.
- ``rig.run(study, **settings)`` runs on a copy with the settings laid over the stored ones; the rig's stored
  settings are unchanged (store them with ``rig.factors['setting:<study>.<key>'] = value``).
- ``rig.robots``, ``rig.sensors`` and ``rig.variants`` are the project's own objects: changing one changes the rig.
- ``rig.add_variant(name, {sensor factors})`` adds a copy of the active variant to ``rig`` (the active one stays).

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
from typing import TYPE_CHECKING, Callable, Sequence

from .core import jobs
from .core.jobs import StudyError

DexVantageError = StudyError

if TYPE_CHECKING:                                    # names for the type hints only (nothing imported at run time)
    import datetime

    import pandas as pd

    from .core.datasets.episode import Dataset, Episode
    from .core.project import Project, Robot, Variant
    from .core.tracking.store import RunStore

__all__ = ['Rig', 'Result', 'Records', 'Table', 'MetricTable', 'Recording', 'DexVantageError', 'open', 'sample',
           'new', 'samples', 'studies', 'metrics', 'exporters', 'plugins', 'load_plugins', 'datasets', 'runs', 'track',
           'run_store', 'publication', 'campaign', 'cells',
           'cluster', 'stats', 'robot_io', 'prices', 'price',
           'diff', 'merge', 'bibtex', 'figures', 'figure', 'export_runs', 'Diff', 'Merge', 'MergeConflict']

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


def _frame(rows: list, columns: list | None = None):
    """A pandas DataFrame of the row dicts when pandas is installed, else Records."""
    rows = [_flatten(r) if isinstance(r, dict) else {'value': r} for r in rows]
    if columns is not None:
        rows = [{c: r.get(c) for c in columns} for r in rows]
    try:
        import pandas as pd
    except ImportError:
        return Records(rows)
    return pd.DataFrame(rows, columns=list(columns) if columns is not None else None)


def _cell(v: object) -> str:
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


def _html_table(rows: list, columns: list | None = None) -> str:
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

    def to_frame(self) -> Records | pd.DataFrame:
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

    def to_frame(self) -> Records | pd.DataFrame:
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


def _fmt(v: object, unit: str) -> str:
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
    if unit == 'p':                                    # a p-value: "p = 0.0915", not "0.0915 p"
        return f'p = {t}'
    if t in ('1', '-1') and len(unit) > 3 and unit.isalpha() and unit.endswith('s'):
        unit = unit[:-1]                               # a counted word: "1 token", not "1 tokens"
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

    def __init__(self, doc: dict, study: str = '', title: str = '', project: Project | None = None):
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
            unit = str(own.get('unit') or '')
            fig = shown
            if isinstance(shown, (int, float)) and not isinstance(shown, bool) and math.isfinite(shown):
                fig = _fmt(shown, unit)                # a number with its unit, as a headline metric shows it
                if isinstance(own.get('quantile'), str) and own['quantile']:
                    fig = f'{own["quantile"]} {fig}'
            text = f'{self.title}: ' + (f'{fig} ({word})' if shown not in (None, '') else word)
            if own.get('text'):
                text += f'. {own["text"]}'
            return {'metric': None, 'title': self.title, 'value': shown, 'unit': unit, 'better': None,
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

    def to_frame(self, key: str | None = None, columns: list | None = None) -> Records | pd.DataFrame:
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

    def __setitem__(self, path: str, value: object):
        self.update({path: value})

    def update(self, values: dict, allow_locked: bool = False) -> None:
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

    def __init__(self, project: Project):
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
    def project(self) -> Project:
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
    def variant(self) -> Variant:
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

    def add_variant(self, name: str, factors: dict | None = None, from_variant: str | None = None,
                    allow_locked: bool = False) -> Variant:
        """Add a variant to this rig (edits it, as the window's "New variant from this one"): a copy of from_variant
        (a name or uid; default the active one) called name, with the sensor factors given set on the copy, e.g.
        rig.add_variant('Raised overhead', {'sensor:cam_high (overhead).dz': 0.05}). Its sensors keep their uids, so
        host links, cables and policy cameras apply to it too; the active variant does not change. Returns the
        Variant (dv.stats.compare_variants(rig, b=name) compares the two)."""
        from .core import factors as F
        from .core.project import Variant, new_uid
        name = str(name or '').strip()
        if not name:
            raise DexVantageError('Give the new variant a name, e.g. rig.add_variant("Raised overhead").')
        if any(x.name == name for x in self._p.variants):
            raise DexVantageError(f'The rig already has a variant called "{name}": choose another name.')
        src = self._p.variant if from_variant is None else (
            self._p.variant_by_uid(from_variant) or next((x for x in self._p.variants if x.name == from_variant),
                                                         None))
        if src is None:
            raise DexVantageError(f'The rig has no variant "{from_variant}" (it has: '
                                  f'{", ".join(x.name for x in self._p.variants)}).')
        values = dict(factors or {})
        other = [k for k in values if not str(k).startswith('sensor:')]
        if other:
            raise DexVantageError(f'{other[0]} is not a sensor factor: a variant holds sensors and their mounts, so '
                                  'only sensor:<name>.<field> factors set a variant (settings belong to the whole '
                                  'project: rig.factors[...]).')
        v = Variant.from_dict(src.to_dict())
        v.uid, v.name, v.note = new_uid(), name, f'A copy of {src.name}'
        p = self._p.snapshot()
        p.variants.append(v)
        if values:
            p = F.apply(p, values, variant=v.uid, allow_locked=allow_locked)    # FactorError: a sentence
            v = p.variant_by_uid(v.uid)
        self._p.variants.append(v)
        return v

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

    def to_frame(self) -> Records | pd.DataFrame:
        """The active variant's sensors, one row each (mount pose in the parent frame)."""
        return _frame(self.sensor_rows())

    # ---------------------------------------------------------------- studies and figures
    def run(self, study: str, *, progress: Callable[[float, str], None] | None = None, **settings: object) -> Result:
        """Run a study (its id, e.g. 'coverage', or its batch flag, e.g. 'grasp') on a copy of the rig with
        settings laid over the stored ones; the rig is unchanged. progress: fn(fraction, text). A setting name the
        study does not read is refused in one sentence that lists its settings (core/study_settings.py)."""
        sid = _study_id(study)
        p = self._p.snapshot()
        if settings:
            from .core import study_settings as SST
            SST.check(sid, settings)                          # an unknown name is refused, never ignored
            st = dict(p.analysis.studies.get(sid) or {})
            st.update(settings)
            p.analysis.studies[sid] = st
        doc = jobs.run_study(sid, p, ctx=jobs.Ctx(progress=progress), cache=_CACHE)
        _autolog(sid, doc, p)                                # core/tracking, when the project's tracking is on
        return Result(doc, sid, project=p)

    def metrics(self, ids: str | list) -> MetricTable:
        """The named metrics (core/metrics.py ids; dv.metrics() lists them), each study run at most once."""
        from .core import metrics as M
        _plugins()
        if isinstance(ids, str):
            ids = [ids]
        return MetricTable(M.evaluate(self._p.snapshot(), list(ids), cache=_CACHE))

    def export(self, out_dir: str, formats: str | list, options: dict | None = None) -> dict:
        """Write the rig in the given formats (dv.exporters() lists them) into out_dir; returns the manifest
        (formats with their files, notes and status; every file's SHA-256; export_manifest.json beside them)."""
        from .core import exporters as EX
        _plugins()
        if isinstance(formats, str):
            formats = [f.strip() for f in formats.split(',') if f.strip()]
        return EX.export(self._p.snapshot(), os.fspath(out_dir), list(formats), options=options)

    def check(self, requirements: str | os.PathLike | list | None = None, variant: str | None = None) -> Result:
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
        _autolog('requirements', {'schema': RQ.SCHEMA, 'study': 'requirements', **card}, self._p, kind='check')
        return Result({'schema': RQ.SCHEMA, 'study': 'requirements', **card}, 'requirements',
                      'Readiness scorecard', project=self._p)

    def synth_episodes(self, folder: str, progress: Callable[[float, str], None] | None = None,
                       **settings: object) -> dict:
        """Render synthetic LeRobot episodes with known faults into a new or empty folder (core/sim/episodes.py;
        needs MuJoCo's offscreen renderer); returns its summary."""
        from .core.sim import episodes as E
        return E.generate(self._p.snapshot(), os.fspath(folder), settings or None, progress=progress)

    def sim2real(self, dataset: str | os.PathLike, *, progress: Callable[[float, str], None] | None = None,
                 **settings: object) -> Result:
        """The sim-to-real gap of a recorded episode (core/sim2real/gap.py: the episode rendered again through the rig
        and compared per camera, the causes ranked with their changes); dataset is a path (LeRobot, RLDS, ROS 2 bag).
        The same as run('sim2real', dataset=path, ...)."""
        return self.run('sim2real', progress=progress, dataset=os.fspath(dataset), **settings)

    def apply_sim2real(self, change: dict) -> 'Rig':
        """A NEW Rig with one sim-to-real suggestion made (a result's 'apply', or a cause's 'apply'); rig is
        unchanged. A locked sensor is refused (DexVantageError)."""
        from .core.sim2real import gap as G
        p = self._p.snapshot()
        G.apply_change(p, change)
        return Rig(p)

    def gap_dataset(self, folder: str, gaps: dict, progress: Callable[[float, str], None] | None = None,
                    **settings: object) -> dict:
        """Render a synthetic LeRobot dataset with KNOWN sim-to-real gaps (brightness_dn, gain, blur_px, noise_dn,
        mount, latency_s, camera: core/sim2real/inject.py) into a new or empty folder; returns its summary."""
        from .core.sim2real import inject as IJ
        return IJ.make_gap_dataset(self._p.snapshot(), os.fspath(folder), gaps, settings or None, progress=progress)

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

    def __init__(self, ds: Dataset):
        self.raw = ds

    format = property(lambda self: self.raw.format)
    version = property(lambda self: self.raw.version)
    path = property(lambda self: self.raw.path)
    info = property(lambda self: self.raw.info)

    def episodes(self) -> Table:
        return Table([{'index': e.index, 'id': e.id, 'length': e.length} for e in self.raw.episodes()])

    def episode(self, index: int) -> Episode:
        return self.raw.load(index)

    def truth(self) -> dict | None:
        return self.raw.truth()

    def episode_rows(self, index: int) -> list:
        """One episode's joint-state rows: {'t', <joint name>: value, ...} per row."""
        ep = self.raw.load(index)
        names = list(ep.joint_names) or [f'q{i}' for i in range(ep.q.shape[1] if ep.q.ndim == 2 else 0)]
        return [{'t': float(t), **{n: float(v) for n, v in zip(names, row)}} for t, row in zip(ep.t, ep.q)]

    def to_frame(self, episode: int | None = None) -> Records | pd.DataFrame:
        """The episodes (one row each), or, with episode=i, that episode's joint-state rows."""
        return _frame(list(self.episodes()) if episode is None else self.episode_rows(episode))

    def close(self) -> None:
        self.raw.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc: object):
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
from .api_cluster import cluster  # noqa: E402 — dv.cluster (api_cluster.py, core/cluster)
from .api_research import (Diff, Merge, MergeConflict, bibtex, diff, export_runs, figure,  # noqa: E402, F401
                           figures, merge)                 # api_research.py


class _Publication:
    """dv.publication: the publication kit (core/publication.py) of a rig, headless.

        dv.publication.methods(rig, results)          the methods paragraph (TeX), every number traced
        dv.publication.tables(rig, results)           {name: TeX table} (sensors, results, scorecard, ...)
        dv.publication.kit(rig, results)              the whole kit as data, with its checks and status
        dv.publication.export(rig, 'out/', results)   main.tex, methods.tex, tables/, figures/, refs.bib, ...

    results: {study id: result} or a list of results (rig.run(...) gives them); without results every study but the
    long meta-studies runs first, as the report does (pass studies=[...] to choose). Other keyword options are the
    kit's settings: sections, siunitx, widths, pdf_figures, bibliography, title, validation, figure_specs. The PDF
    figures (export only) use Qt's PDF writer; nothing else imports Qt."""

    @staticmethod
    def _opts(rig: Rig | Project, results: dict | list | None, options: dict):
        from .core.project import Project
        p = rig.project if isinstance(rig, Rig) else rig
        if not isinstance(p, Project):
            raise DexVantageError('dv.publication takes a Rig (dv.open, dv.sample) or a Project.')
        o = dict(options)
        if results is not None:
            o['results'] = {k: dict(v) for k, v in results.items()} if isinstance(results, dict) else \
                [dict(r) for r in results]
            o.setdefault('studies', [])
        return p, o

    def kit(self, rig: Rig | Project, results: dict | list | None = None, **options: object) -> dict:
        from .core import publication as PB
        _plugins()
        p, o = self._opts(rig, results, options)
        k = PB.build(p, o)
        out = PB.public(k)
        out.update(parts=dict(k['parts']), bib=k['bib'])
        return out

    def methods(self, rig: Rig | Project, results: dict | list | None = None, **options: object) -> str:
        options['sections'] = ['methods']
        return self.kit(rig, results, **options)['parts'].get('methods', '')

    def tables(self, rig: Rig | Project, results: dict | list | None = None, **options: object) -> dict:
        parts = self.kit(rig, results, **options)['parts']
        return {k: v for k, v in parts.items() if k not in ('methods', 'appendix')}

    def export(self, rig: Rig | Project, out_dir: str, results: dict | list | None = None, **options: object) -> dict:
        from .core import exporters as EX
        _plugins()
        p, o = self._opts(rig, results, options)
        return EX.export(p, os.fspath(out_dir), ['paper_kit'], {'paper_kit': o})

    def __repr__(self) -> str:
        return '<dv.publication: methods(rig), tables(rig), kit(rig), export(rig, out_dir)>'


publication = _Publication()


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


# ====================================================================== cited prices (core/prices.py)
def prices(kind: str | None = None, stated_only: bool = True) -> Table:
    """The library's cited prices: id, kind, device, amount, currency, price kind (maker_store, maker_price_list,
    reseller), min_quantity, region, source URL and date read; stated_only=False lists the entries without one too
    (amount None). Nothing is estimated or converted (docs/models/library.md#prices)."""
    from .core import library as LB
    from .core import prices as PR
    rows = []
    for e in LB.entries(kind):
        p = PR.block(e) or {}
        a, cur = PR.stated(e)
        if stated_only and a is None and not p.get('options'):
            continue
        rows.append({'id': e.id, 'kind': e.kind, 'device': e.title, 'amount': a, 'currency': cur or p.get('currency'),
                     'price_kind': p.get('kind'), 'min_quantity': p.get('min_quantity'), 'region': p.get('region'),
                     'source': p.get('source'), 'read': p.get('read'), 'options': len(p.get('options') or []),
                     'custom': e.is_custom})
    return Table(rows)


def price(model_id: str) -> dict | None:
    """The cited price block of one library entry (a copy), or None when the entry states no price."""
    import copy
    from .core import library as LB
    from .core import prices as PR
    e = LB.get(model_id)
    if e is None:
        raise StudyError(f'There is no library entry called {model_id!r}.')
    p = PR.block(e)
    return copy.deepcopy(p) if p is not None else None


# ====================================================================== the experiment tracker (core/tracking)
def _autolog(study: str, doc: dict, project: Project, kind: str | None = None):
    """Log a run when the project's tracking is on (a warning, never an exception, when the store cannot be
    written: the result itself is fine)."""
    from .core import tracking as TR
    if not TR.should_log(study, project):
        return
    try:
        TR.log(study, doc, project, kind=kind, interface='api')
    except TR.TrackingError as e:
        import warnings
        warnings.warn(f'The run was not logged: {e}', stacklevel=3)


def run_store(where: Rig | str | os.PathLike | RunStore | None) -> RunStore:
    """The run store (core/tracking/store.py RunStore) of where: a Rig (its project's store), a .dexv project file
    (the store its tracking settings name, default <name>.runs beside it) or a run store folder."""
    from .core import tracking as TR
    if isinstance(where, Rig):
        return TR.store_for(where.project)
    if isinstance(where, TR.RunStore):
        return where
    if where is None:
        raise DexVantageError('Name the run store: dv.runs(rig), dv.runs("cell.dexv") or dv.runs("cell.runs").')
    path = os.fspath(where)
    if os.path.isfile(path) or path.lower().endswith('.dexv'):
        return TR.store_for(Rig.open(path).project)
    return TR.RunStore(path)


def runs(where: Rig | str | os.PathLike | RunStore | None = None, *, study: str | list | None = None,
         tag: str | list | None = None, kind: str | list | None = None, since: str | datetime.date | None = None,
         until: str | datetime.date | None = None, status: str | list | None = None,
         project: str | list | None = None) -> Records | pd.DataFrame:
    """The runs of a run store (run_store(where)) as a table: one row per run (run id, start and end in UTC,
    seconds, kind, study, status, project, its content hash, variant, headline, tags, note, re-run of, DexVantage
    version, git commit, result hash) and one column per metric (a metric id, or id@stat for a distribution's or a
    search's figure); None where a run does not state it. A pandas DataFrame (units in .attrs['units']) when pandas
    is installed, else Records (a list of row dicts, with .units). Filters: study, tag, kind, status, project (a
    value or a list), since / until (a date, a datetime or ISO text)."""
    from .core.tracking import tables as TB
    st = run_store(where)
    if not st.exists():
        raise DexVantageError(f'There is no run store at {st.name} yet: turn the project\'s tracking on '
                              '(rig.factors["setting:tracking.enabled"] = True, or the Runs page) and run a study, or '
                              'log a result with dv.track(result, rig).')
    rows = st.runs(study=study, tag=tag, kind=kind, since=since, until=until, status=status, project=project)
    cols, data, units = TB.wide(rows)
    try:
        import pandas as pd
    except ImportError:
        out = Records(data)
        out.units = units
        return out
    df = pd.DataFrame(data, columns=cols)
    df.attrs['units'] = units
    df.attrs['store'] = st.name
    return df


def track(result: Result | dict, rig: Rig | None = None, *, store: str | os.PathLike | RunStore | None = None,
          tags: tuple | list = (), note: str = '') -> dict:
    """Log a Result (from rig.run or rig.check) as a run, whether or not the project's tracking is on: into store
    (a folder or a RunStore; default the rig's project's store). Returns the run record."""
    from .core import tracking as TR
    if not isinstance(result, dict) or not result.get('study'):
        raise DexVantageError('dv.track takes a result of rig.run(...) or rig.check(...).')
    project = rig.project if isinstance(rig, Rig) else getattr(result, '_project', None)
    if project is None:
        raise DexVantageError('dv.track needs the rig the result was made from: dv.track(result, rig).')
    st = run_store(store) if store is not None else TR.store_for(project)
    sid = result['study']
    return TR.log(sid, dict(result), project, store=st, kind='check' if sid == 'requirements' else None,
                  interface='api', tags=tags, note=note)


# ====================================================================== the campaign planner (core/campaign.py)
class _Campaign:
    """dv.campaign: plan(rig, spec=None, **settings) -> Result (the 'campaign' study: the counterbalanced schedule,
    totals, storage, checklists); export(result, base) -> the paths of <base>.csv, .ics, .html; power_n(p1, p2,
    alpha=0.05, power=0.8, continuity=True) -> the demonstrations per condition for a difference in success rate;
    balance(result) -> the balance counts; apply(rig, result) writes the plan into the rig's project."""

    def plan(self, rig: Rig | Project, spec: dict | str | None = None, **settings: object) -> Result:
        """The campaign of rig (a Rig or a core Project) with spec (a dict, JSON text or a spec file) and settings
        laid over the project's stored spec; the stored spec is not changed."""
        from .core import campaign as CP
        p = rig.project if isinstance(rig, Rig) else rig
        snap = p.snapshot()
        stored = {k: v for k, v in (snap.analysis.studies.get(CP.STUDY_ID) or {}).items() if k != 'plan'}
        extra = CP.read_spec(spec) if spec is not None else {}
        snap.analysis.studies[CP.STUDY_ID] = {**stored, **extra, **(CP.read_spec(settings) if settings else {})}
        doc = jobs.run_study(CP.STUDY_ID, snap, cache=_CACHE)
        _autolog(CP.STUDY_ID, doc, snap)
        res = Result(doc, CP.STUDY_ID, project=snap)
        res.spec = dict(snap.analysis.studies[CP.STUDY_ID])          # what apply() stores with the plan
        return res

    def export(self, result: Result | dict, base: str, start_date: str | None = None) -> list:
        from .core import campaign as CP
        return CP.write_files(dict(result), os.fspath(base), start_date)

    def power_n(self, p1: float, p2: float, alpha: float = 0.05, power: float = 0.8, continuity: bool = True) -> dict:
        from .core import campaign as CP
        return CP.power_n(p1, p2, alpha, power, continuity)

    def balance(self, result: Result | dict) -> dict:
        from .core import campaign as CP
        return CP.balance(dict(result))

    def apply(self, rig: Rig | Project, result: Result | dict) -> None:
        from .core import campaign as CP
        p = rig.project if isinstance(rig, Rig) else rig
        CP.apply_plan(p, dict(result), getattr(result, 'spec', None))

    def __repr__(self) -> str:
        return '<dv.campaign: plan(rig, spec), export(result, base), power_n(p1, p2), balance(result), apply(rig, ' \
               'result)>'


campaign = _Campaign()


# ====================================================================== multi-cell comparison (core/cells.py)
class CellComparison(Result):
    """The comparison of several cells (dv.cells.compare): the 'cells' study's document plus

        .pooling_mapping()    the pooling-mapping document (dexvantage.cells_pooling/1)
        .metric_rows()        one row per metric, a column per cell (to_frame-ready)
        .gap_rows()           one row per matched camera pair against the reference
        .save_files(out_dir)  cells.json, cells_pooling.json, cells_comparison.html (with provenance)
        .html()               the comparison as a standalone HTML page
    """

    def pooling_mapping(self) -> dict:
        from .core import cells as CE
        return CE.pooling_mapping(dict(self))

    def metric_rows(self) -> list:
        names = [c['name'] for c in self['cells']]
        return [{'metric': r['id'], 'title': r['title'], 'unit': r['unit'], 'better': r['better'],
                 **{n: v for n, v in zip(names, r['values'])},
                 'best': names[r['best']] if r['best'] is not None else None,
                 'worst': names[r['worst']] if r['worst'] is not None else None}
                for r in self['metrics']['rows']]

    def gap_rows(self) -> list:
        return [{'cell': g['b'], 'gap': g['total'], 'band': g['band'], **{k: m[k] for k in (
            'a', 'b', 'role', 'viewpoint_deg', 'rotation_deg', 'distance_m', 'hfov_diff_deg', 'resolution_ratio',
            'image_pct_a', 'image_pct_b')}} for g in self['gap']['pairs'] for m in g['matches']]

    def save_files(self, out_dir: str) -> list:
        from .core import cells as CE
        return [os.path.join(os.fspath(out_dir), n) for n in CE.write_files(dict(self), self._project,
                                                                            os.fspath(out_dir))]

    def html(self) -> str:
        from .core import cells as CE
        return CE.to_html(dict(self))


class _Cells:
    """dv.cells: compare(*cells, reference=0, metrics=None, tolerance_draws=0, seed=1, pooling='auto',
    variants=False, progress=None) -> CellComparison.

    A cell is a Rig (dv.open, dv.sample), a core Project, a path to a .dexv file or folder, or 'sample:<name>'; the
    first is the reference unless reference= names another (by number). metrics: metric ids, 'all', or None (the
    quick figures, core/cells.DEFAULT_METRICS); tolerance_draws ≥ 2 runs the tolerance Monte Carlo on every cell to say
    which differences exceed the build spread; pooling: 'auto' (a recorded dataset where a cell has one, else its
    plan), 'plan' or 'recorded'; variants=True adds the first cell's other variants as cells.

    >>> import dexvantage as dv
    >>> res = dv.cells.compare(dv.sample('so100'), 'sample:so100', metrics=['rig.camera_count'])
    >>> res.headline['value'], res['pooling']['text']
    (0.0, 'Every cell pools as is.')
    """

    def compare(self, *cells: Rig | Project | str | os.PathLike, reference: int = 0, metrics: list | None = None,
                tolerance_draws: int = 0, seed: int = 1, pooling: str = 'auto', variants: bool = False,
                progress: Callable[[float, str], None] | None = None) -> CellComparison:
        from .core import cells as CE
        items = []
        for c in cells:
            items += list(c) if isinstance(c, (list, tuple)) else [c]
        if not items:
            raise DexVantageError('dv.cells.compare takes two or more cells: Rigs, projects, .dexv paths or '
                                  '"sample:<name>".')
        first = CE.make_cells(items[:1])[0]
        cs = CE.make_cells(items[1:], open_project=first.project, variants=variants)
        cs[0].source, cs[0].label = first.source, first.label
        ctx = jobs.Ctx(progress=progress)
        doc = CE.jsonable(CE.compare(cs, reference, metrics, tolerance_draws, seed, pooling, ctx))
        _autolog(CE.SID, {'study': CE.SID, **doc}, cs[0].project)     # the first cell's tracking, when on
        return CellComparison({'study': CE.SID, **doc}, CE.SID, project=cs[0].project)

    def __repr__(self) -> str:
        return '<dv.cells: compare(*cells, reference=0, metrics=None, tolerance_draws=0, pooling="auto")>'


cells = _Cells()


# ====================================================================== statistics and power (core/stats.py)
class _Stats:
    """dv.stats: tests, effect sizes, corrections and power analysis (core/stats.py, every function of it is here:
    dv.stats.wilson, .clopper_pearson, .bootstrap_ci, .permutation_test, .welch_t, .mann_whitney,
    .wilcoxon_signed_rank, .cohens_d, .cliffs_delta, .risk_difference, .odds_ratio, .fisher_exact, .adjust,
    .power_two_proportions, .power_t, .n_two_proportions, .n_t, .episodes_needed, .demonstrations_needed, ...) and the
    app's comparisons:

        compare_rates(rows, compare='reference', test='fisher', ...)     -> Result (the 'stats' study)
        compare_samples(a, b, paired=False, title='', unit='', better='higher', ...) -> Result
        compare_variants(rig, a=None, b=None, metrics=None, n=32, seed=0, ...)       -> Result (tolerance draws,
                                                                                         paired by seed)
        compare_experiment(rig, factor, levels=None, result=None, ...)              -> Result
        compare_runs(rig, a, b, folder=None, metrics=None, ...)                     -> Result (tracker runs)
        run(spec, rig=None)                                                         -> Result (a spec dict or file)

    >>> import dexvantage as dv
    >>> dv.stats.episodes_needed(0.5, 0.8)['n_per_group']
    44
    >>> r = dv.stats.compare_rates([('ACT', 14, 20), ('Diffusion', 19, 20)])
    >>> r.headline['word'], round(r['comparisons'][0]['p'], 4)
    ('no clear difference', 0.0915)
    """

    def __getattr__(self, name: str):
        if name.startswith('_'):
            raise AttributeError(name)
        from .core import stats as ST
        try:
            return getattr(ST, name)
        except AttributeError:
            raise AttributeError(f'dv.stats has no {name}') from None

    def __dir__(self):
        from .core import stats as ST
        return sorted(set(dir(type(self))) | {k for k in dir(ST) if not k.startswith('_')})

    @staticmethod
    def _run(spec: dict, rig: Rig | Project | None = None,
             progress: Callable[[float, str], None] | None = None) -> Result:
        from .core.studies import stats as SS
        p = rig.project if isinstance(rig, Rig) else rig
        doc, snap = SS.run_spec(spec, p, jobs.Ctx(progress=progress))
        if snap is not None and p is not None:
            _autolog(SS.SID, doc, snap)                      # the rig's tracking, when on (rates alone have no rig)
        return Result(doc, SS.SID, project=snap)

    def run(self, spec: dict | str | os.PathLike, rig: Rig | Project | None = None,
            progress: Callable[[float, str], None] | None = None) -> Result:
        """A statistics spec (a dict, JSON text or a spec file: schema dexvantage.stats/1) on rig (None: success
        counts, samples and the power calculator need no project)."""
        from .core.studies import stats as SS
        return self._run(SS.read_spec(spec), rig, progress)

    def compare_rates(self, rows: list, compare: str = 'reference', test: str = 'fisher', power: dict | None = None,
                      **settings: object) -> Result:
        """rows: [(condition, successes, trials)] or [{'condition', 'successes', 'trials'}]."""
        rr = [r if isinstance(r, dict) else {'condition': r[0], 'successes': r[1], 'trials': r[2]} for r in rows]
        spec = {'source': 'rates', 'rates': rr, 'compare': compare, 'rate_test': test, **settings}
        if power:
            spec['power'] = power
        return self._run(spec)

    def compare_samples(self, a: Sequence[float], b: Sequence[float], paired: bool = False, title: str = 'Value',
                        unit: str = '', better: str = 'higher', names: tuple | list = ('A', 'B'),
                        **settings: object) -> Result:
        return self._run({'source': 'samples', 'samples': {'a': list(a), 'b': list(b), 'paired': paired,
                                                           'title': title, 'unit': unit, 'better': better,
                                                           'names': list(names)}, **settings})

    def compare_variants(self, rig: Rig | Project, a: str | None = None, b: str | None = None,
                         metrics: list | None = None, n: int = 32, seed: int = 0, scale: float = 1.0,
                         distribution: str = 'uniform', progress: Callable[[float, str], None] | None = None,
                         **settings: object) -> Result:
        """Two variants of rig over the tolerance Monte Carlo with one seed (the draws paired by seed)."""
        v = {'a': a, 'b': b, 'metrics': list(metrics or []), 'n': n, 'seed': seed, 'scale': scale,
             'distribution': distribution}
        return self._run({'source': 'variants', 'variants': {k: x for k, x in v.items() if x not in (None, '')},
                          **settings}, rig, progress)

    def compare_experiment(self, rig: Rig | Project, factor: str, levels: list | None = None,
                           result: Result | dict | str | os.PathLike | None = None, responses: list | None = None,
                           progress: Callable[[float, str], None] | None = None, **settings: object) -> Result:
        """An experiment's runs split by factor's two levels (result: an experiment Result, its dict or a file;
        default the experiment saved in rig, run)."""
        ex = {'mode': 'factor', 'factor': factor}
        if levels is not None:
            ex['levels'] = list(levels)
        if result is not None:
            ex['result'] = dict(result) if isinstance(result, dict) else os.fspath(result)
        if responses:
            ex['responses'] = list(responses)
        return self._run({'source': 'experiments', 'experiment': ex, **settings}, rig, progress)

    def compare_runs(self, rig: Rig | Project, a: str | list | dict, b: str | list | dict, folder: str | None = None,
                     metrics: list | None = None, **settings: object) -> Result:
        """Tracker runs: a and b each a run id, a list of ids or {'tag': t}."""
        tr = {'a': a, 'b': b, 'folder': os.fspath(folder) if folder else '', 'metrics': list(metrics or [])}
        return self._run({'source': 'tracker', 'tracker': tr, **settings}, rig)

    def __repr__(self) -> str:
        return ('<dv.stats: compare_rates, compare_samples, compare_variants, compare_experiment, compare_runs, run; '
                'and core/stats (wilson, bootstrap_ci, permutation_test, adjust, episodes_needed, ...)>')


stats = _Stats()


class _RobotIO:
    """dv.robot_io: robot files in and out, headless.

        dv.robot_io.convert_usd('arm.usda')          # {'name', 'mjcf', 'meshes', 'notes', 'record'}: a UsdPhysics
                                                     # articulation as MJCF (core/robot/usd_import.py; needs usd-core)
        robot = dv.robot_io.add_robot(rig, 'arm.usda')   # import a URDF, MJCF or USD robot into rig (edits rig)
        dv.robot_io.write_usd(rig, "arm.usda")       # a robot as a UsdPhysics articulation (core/robot/usd_writer)
        dv.robot_io.openvla_encoding(rig)            # the OXE state / action encodings the OpenVLA export writes
    """

    def convert_usd(self, path: str | os.PathLike) -> dict:
        from .core.robot import usd_import as U
        try:
            c = U.convert(os.fspath(path))
        except U.UsdImportError as e:
            raise DexVantageError(str(e)) from None
        return {'name': c.name, 'mjcf': c.mjcf, 'meshes': sorted(c.meshes), 'notes': list(c.notes),
                'record': dict(c.record)}

    def add_robot(self, rig: Rig, path: str | os.PathLike, name: str | None = None) -> Robot:
        """Import the robot file at path (URDF, MJCF or USD) into rig's project (the first robot is primary); the
        Robot added. Edits rig."""
        from .core.project import ProjectFormatError
        from .core.robot import import_robot
        try:
            robot, sources = import_robot(os.fspath(path), name)
        except ProjectFormatError as e:
            raise DexVantageError(str(e)) from None
        p = rig.project
        robot.primary = not p.robots
        p.robots.append(robot)
        p.sources.update(sources)
        return robot

    def write_usd(self, rig: Rig, path: str | os.PathLike, robot: str | None = None) -> dict:
        """Write rig's robot (a name or uid; default the primary) as a UsdPhysics articulation at path."""
        from .core.robot.model import RobotModelError
        from .core.robot.usd_import import have_pxr, missing_pxr_message
        from .core.robot.usd_writer import write_robot_usd
        if not have_pxr():
            raise DexVantageError(missing_pxr_message(os.path.basename(os.fspath(path))).replace(
                'is a USD file, which needs', 'is a USD file: writing it needs'))
        rs = rig.project.robots
        r = next((x for x in rs if robot in (x.name, x.uid)), None) if robot else (rig.project.primary_robot or
                                                                                  (rs[0] if rs else None))
        if r is None:
            raise DexVantageError('The rig has no such robot.' if robot else 'The rig has no robot.')
        try:
            return write_robot_usd(rig.project, r.uid, os.fspath(path))
        except RobotModelError as e:
            raise DexVantageError(str(e)) from None

    def openvla_encoding(self, rig: Rig, action_space: str | None = None) -> dict:
        from .core.exporters.lerobot import openvla_encoding
        return openvla_encoding(rig.project, action_space)

    def __repr__(self) -> str:
        return '<dv.robot_io: convert_usd, add_robot, write_usd, openvla_encoding>'


robot_io = _RobotIO()
