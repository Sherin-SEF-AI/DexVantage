"""The research tools of part 1 from Python: project diff and merge, BibTeX and publication figures (dv.diff, dv.merge,
dv.bibtex, dv.figures, dv.figure). Batch mode had them (--diff, --merge, --cite, --figure; batch_research.py); a
notebook had to import core modules for them. These are thin layers over the same code (core/projdiff.py,
core/cite.py, core/figures.py and the paper kit's figure specs, core/publication.figure_specs): they compute nothing
of their own.

    d = dv.diff(rig, other)                       a Diff: the change list (a dict, schema dexvantage.diff/1) with
                                                  .identical, .text, .html, .to_frame(), .save(path) and an HTML view
    d = dv.diff('a.dexv', 'b.dexv', metrics=['coverage.covered'])   also run these metrics on both and give deltas
    m = dv.merge(base, ours, theirs)              a Merge: .rig (the merged Rig), .from_ours, .from_theirs, .notes;
                                                  conflicts raise dv.MergeConflict (a DexVantageError, .conflicts)
    dv.bibtex(rig)                                BibTeX of every source the rig depends on (path= also writes it)
    dv.figures(result)                            the publication figure specs of a result (the paper kit's)
    dv.figure(result or spec, 'fig.svg')          a vector figure (SVG; PDF by the extension, through Qt's PDF writer)
    dv.export_runs(rig or store, 'out/')          the tracked runs as CSV, Parquet, an MLflow mlruns/ folder and a
                                                  W&B offline replay (core/tracking.export, as --runs-export)

Projects are given as a Rig, a core Project, or the path of a .dexv file or unpacked .dexv.d folder. No Qt is
imported on this path except to write a PDF figure.

Known limits: as the modules they call (lists such as grasp poses and joint tables are diffed and merged whole; only
the curve and bar charts have figure specs, so a result without a per-item list of numbers has no figure); the
metric deltas of dv.diff run each named metric's study on both projects in this thread.
"""
from __future__ import annotations

import builtins
import json
import os
from typing import TYPE_CHECKING, Sequence

from .core import jobs
from .core.projdiff import MergeConflict

if TYPE_CHECKING:                                    # names for the type hints only (nothing imported at run time)
    import pandas as pd

    from .api import Records, Result, Rig
    from .core.project import Project
    from .core.projdiff import MergeResult
    from .core.tracking.store import RunStore

DexVantageError = jobs.StudyError

__all__ = ['Diff', 'Merge', 'MergeConflict', 'diff', 'merge', 'bibtex', 'figures', 'figure', 'export_runs']


def _project(x: Rig | Project | str | os.PathLike, what: str = 'project') -> Project:
    """A core Project from a Rig, a Project or a path (DexVantageError in one sentence otherwise)."""
    from .core.project import Project, ProjectFormatError
    p = getattr(x, 'project', None)
    if isinstance(p, Project):
        return p
    if isinstance(x, Project):
        return x
    if isinstance(x, (str, os.PathLike)):
        try:
            return Project.load(os.fspath(x))
        except ProjectFormatError as e:
            raise DexVantageError(str(e)) from None
    raise DexVantageError(f'The {what} is a Rig (dv.open, dv.sample), a Project or the path of a .dexv file or '
                          f'unpacked folder; it was {type(x).__name__}.')


# ====================================================================== diff
class Diff(dict):
    """What changed from project a to project b (core/projdiff.diff's document): 'a', 'b', 'identical', 'counts',
    'changes' (section, op, kind, item, field, path, before, after, text) and 'metrics' (the deltas asked for)."""

    def __init__(self, doc: dict, project: Project | None = None):
        super().__init__(doc)
        self._project = project

    @property
    def identical(self) -> bool:
        return bool(self.get('identical'))

    @property
    def text(self) -> str:
        from .core import projdiff as D
        return D.to_text(dict(self))

    @property
    def html(self) -> str:
        """A standalone HTML page of the diff."""
        from .core import projdiff as D
        return D.to_html(dict(self))

    def to_frame(self, key: str = 'changes') -> Records | pd.DataFrame:
        """One row per change (key='metrics': one row per metric delta): a pandas DataFrame, or Records."""
        from .api import _frame
        if key not in ('changes', 'metrics'):
            raise DexVantageError(f'A diff has the tables "changes" and "metrics", not "{key}".')
        cols = (['section', 'op', 'kind', 'item', 'field', 'text'] if key == 'changes' else
                ['id', 'title', 'unit', 'a', 'b', 'delta', 'verdict'])
        return _frame(list(self.get(key) or []), cols)

    def save(self, path: str) -> str:
        """The diff as JSON with the provenance block of project a (as --diff A --diff B OUT.json writes it)."""
        from .core import provenance as PV
        path = os.fspath(path)
        doc = dict(self)
        if self._project is not None:
            doc = PV.stamp(doc, self._project, output={'option': 'diff', 'name': os.path.basename(path),
                                                       'in_folder': False})
        try:
            with builtins.open(path, 'w', encoding='utf-8') as fh:
                json.dump(doc, fh, indent=1, ensure_ascii=False, default=str)
                fh.write('\n')
        except OSError as e:
            raise DexVantageError(f'Could not write {os.path.basename(path)}: {e.strerror or e}.') from None
        return path

    def __repr__(self) -> str:
        n = len(self.get('changes') or [])
        return (f'<Diff {self["a"]["name"]!r} → {self["b"]["name"]!r}: '
                + ('no differences' if self.identical else f'{n} change{"" if n == 1 else "s"}') + '>')

    def _repr_html_(self) -> str:
        page = self.html
        body = page[page.index('<body>') + 6: page.rindex('</body>')]
        return f'<div>{body}</div>'


def diff(a: Rig | Project | str | os.PathLike, b: Rig | Project | str | os.PathLike,
         metrics: str | Sequence[str] | None = None) -> Diff:
    """What changed from a to b (each a Rig, a Project or a path). metrics: metric ids (dv.metrics()) to run on both
    projects and compare; their studies run in this thread."""
    from .core import projdiff as D
    pa, pb = _project(a, 'first project'), _project(b, 'second project')
    doc = D.diff(pa, pb)
    if metrics:
        from .core import metrics as M
        M.load_metrics()
        ids = [metrics] if isinstance(metrics, str) else [str(m) for m in metrics]
        unknown = [m for m in ids if m not in M.METRICS]
        if unknown:
            raise DexVantageError(f'Unknown metric{"s" if len(unknown) > 1 else ""} {", ".join(unknown)}: '
                                  'dv.metrics() lists them.')
        from .api import _CACHE
        doc['metrics'] = D.evaluate_deltas(pa, pb, ids, cache=_CACHE)
    return Diff(doc, pa)


# ====================================================================== merge
class Merge:
    """A three-way merge: .rig (the merged Rig, not yet saved), .from_ours and .from_theirs (the places only one side
    changed), .notes (what the merge noticed, such as two sensors that now share a name)."""

    def __init__(self, res: MergeResult):
        from .api import Rig
        self.rig = Rig(res.project)
        self.from_ours = list(res.from_ours)
        self.from_theirs = list(res.from_theirs)
        self.notes = list(res.notes)

    def __repr__(self) -> str:
        return (f'<Merge: {len(self.from_ours)} change{"" if len(self.from_ours) == 1 else "s"} of ours, '
                f'{len(self.from_theirs)} of theirs{", " + str(len(self.notes)) + " notes" if self.notes else ""}>')


def merge(base: Rig | Project | str | os.PathLike, ours: Rig | Project | str | os.PathLike,
          theirs: Rig | Project | str | os.PathLike) -> Merge:
    """Merge two edits (ours, theirs) of one project (base); each a Rig, a Project or a path. Places changed
    differently on both sides raise dv.MergeConflict (a DexVantageError) whose .conflicts lists them in words."""
    from .core import projdiff as D
    pb = _project(base, 'base project')
    po = _project(ours, 'our project')
    pt = _project(theirs, 'their project')
    return Merge(D.merge3(pb, po, pt))


# ====================================================================== citations
def bibtex(rig: Rig | Project | str | os.PathLike | None, path: str | os.PathLike | None = None,
           references: str = 'checks') -> str:
    """BibTeX of every source the rig depends on: DexVantage, its library entries' datasheets, its robots' model
    files, its datasets and the validation references (references: 'checks' those the checks cite, 'all', 'none').
    path: also write it there. Returns the text."""
    from .core import cite as C
    if references not in ('checks', 'all', 'none'):
        raise DexVantageError(f'references is "checks", "all" or "none", not {references!r}.')
    p = _project(rig, 'rig') if rig is not None else None
    text = C.bibtex(p, references)
    if path:
        path = os.fspath(path)
        try:
            with builtins.open(path, 'w', encoding='utf-8') as fh:
                fh.write(text)
        except OSError as e:
            raise DexVantageError(f'Could not write {os.path.basename(path)}: {e.strerror or e}.') from None
    return text


# ====================================================================== figures
def figures(result: Result | dict) -> list:
    """The publication figure specs of a study result (a dv Result or its dict), as the paper kit draws them: a list
    of {'id', 'study', 'spec'}; empty when the result has no per-item list of numbers to chart."""
    from .core import publication as PB
    if not isinstance(result, dict):
        raise DexVantageError('dv.figures takes a study result (rig.run(...)).')
    sid = getattr(result, 'study', '') or str(result.get('study') or '')
    if not sid:
        raise DexVantageError('The result does not say which study made it (no "study" field).')
    jobs.load_studies()
    return [{'id': fid, 'study': s, 'spec': spec} for fid, spec, s in PB.figure_specs({sid: dict(result)})]


def figure(source: Result | dict | str | os.PathLike, path: str | os.PathLike, width: str | float = 'single',
           palette: str = 'tol', index: int = 0, **options: object) -> dict:
    """Write a vector figure: source is a figure spec (dexvantage.figure/1, a dict, JSON text or a .json file), a
    {'spec': ...} item of dv.figures(), or a study result (its figure number index). width: 'single' (88.9 mm),
    'double' (182 mm) or millimetres; palette 'tol' (colour-blind safe, the default) or 'chart'. The format follows
    the extension (.svg or .pdf). Returns {'path', 'format', 'width_mm', 'height_mm', 'series'}."""
    from .core import figures as F
    spec = None
    if isinstance(source, dict) and source.get('kind') in ('curves', 'bars'):
        spec = source
    elif isinstance(source, dict) and isinstance(source.get('spec'), dict):
        spec = source['spec']
    elif isinstance(source, (str, os.PathLike)):
        spec = F.read_spec(os.fspath(source))
    elif isinstance(source, dict):
        figs = figures(source)
        if not figs:
            title = getattr(source, 'title', '') or str(source.get('study') or 'study')
            raise DexVantageError(f'The {title} result has no chart to draw (no per-item list of numbers).')
        if not 0 <= int(index) < len(figs):
            raise DexVantageError(f'The result has {len(figs)} figure{"" if len(figs) == 1 else "s"}; index '
                                  f'{index} is not one of them.')
        spec = figs[int(index)]['spec']
    else:
        raise DexVantageError('dv.figure takes a figure spec, an item of dv.figures(result) or a study result.')
    if palette not in ('tol', 'chart'):
        raise DexVantageError(f'The palette is "tol" or "chart", not {palette!r}.')
    if isinstance(width, str) and width not in ('single', 'double'):
        try:
            width = float(width)
        except ValueError:
            raise DexVantageError(f'The width is "single", "double" or millimetres, not {width!r}.') from None
    spec = F.check_spec(spec)
    if spec['kind'] == 'bars' and options.get('height_mm') is None:      # sized as the paper kit sizes its bars
        from .core import publication as PB
        options['height_mm'] = max(F.size_mm(width)[1], 10.0 + 5.0 * len(spec.get('rows') or []))
        spec = PB.fit_labels(spec, width)
    return F.write(spec, os.fspath(path), width=width, palette=palette, **options)


# ====================================================================== tracked runs
def export_runs(where: Rig | str | os.PathLike | RunStore | None, out_dir: str | os.PathLike,
                formats: str | Sequence[str] = ('csv', 'parquet', 'mlflow', 'wandb'), **filters: object) -> dict:
    """Write the runs of a run store (dv.run_store(where): a Rig, a .dexv file or a store folder) into out_dir as
    runs.csv / runs_long.csv, runs.parquet, mlruns/ (MLflow's file store layout) and wandb/ (a W&B offline replay),
    with a manifest (SHA-256 and provenance). filters as dv.runs (study, tag, kind, status, project, since, until).
    Returns {'files', 'formats': {format: {'status', 'files', 'notes' or 'reason'}}, 'runs', 'manifest'}."""
    from .api import run_store
    from .core import tracking as TR
    st = run_store(where)
    if not st.exists():
        raise DexVantageError(f'There is no run store at {st.name} yet: log a result with dv.track(result, rig) or '
                              'turn the project\'s tracking on first.')
    try:
        return TR.export(st, os.fspath(out_dir), formats, **filters)
    except OSError as e:
        raise DexVantageError(f'Could not write the runs into {os.path.basename(os.fspath(out_dir))}: '
                              f'{e.strerror or e}.') from None
