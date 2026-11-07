"""The publication kit: a methods paragraph, LaTeX tables, vector figures, a BibTeX file and a reproducibility
appendix written from the project and the study results present, every number traceable to where it came from.

    kit = build(project, options) -> dict                 schema 'dexvantage.paper_kit/1' (JSON-able)
    write(project, out_dir, options, ctx) -> {'files', 'notes', 'kit'}      the files under out_dir/paper_kit/
    methods(project, results, ...)                        the methods paragraph alone (build() calls it)
    numbers_in(tex) -> [token]                            every number written in a TeX text
    verify_numbers(tex, trace, source) -> {'numbers', 'traced', 'problems'}
    cited_keys(tex) / bib_keys(bib) / latex_problems(tex)  the structural checks the page shows

Built on core/cite.py (the BibTeX entries: DexVantage itself, the library sources of every device, the robot model
files, the references of the validation checks), core/figures.py (SVG and PDF figures at IEEE's one- and two-column
widths) and the 'report' exporter (core/exporters/report.py: collect() gathers the results present or runs the
studies asked for, metric_rows() reads each study's named figures, figure() finds a per-camera chart).

**Where each number comes from.** The methods paragraph is written through a Writer that records, for every number
it writes, a trace entry: the place in the source data the number is read from (a path into the project's JSON, a
library entry, a study result, the software environment, the validation report or the check registry) and the text
written. verify_numbers() resolves every path again, checks that the written number is the source value rounded to
the digits shown, and that every number in the paragraph has an entry. Counts ("2 sensors") are the length of a list
in the source data. Units are never converted in the prose: positions are in metres and orientations are the unit
quaternion (x, y, z, w), as the project stores them; numbers are rounded only (3 decimals for metres, 4 for a
quaternion, as stated in DEFAULTS). Each table row ends with a comment "% dv-src: [...]" naming the source of each
number cell (\\num{} with siunitx, else \\dvnum{}), so the tables are checked the same way.

**Tables** use booktabs (\\toprule, \\midrule, \\bottomrule, no vertical or double rules, units in the column
heading where a column has one unit: Fear, "Publication quality tables in LaTeX", booktabs v1.61803398, CTAN, read
2026-11-07). With the setting siunitx (a DexVantage default of False) numbers are written \\num{...} and units
\\unit{...} (siunitx v3, "literal" unit input such as \\unit{Mbit/s}; Wright, siunitx 3.6.7 manual, CTAN, read
2026-11-07); without it \\dvnum{...}, a macro each file defines as math mode, so no package beyond booktabs and
graphicx is needed.

**Validation status** of a study's model: the validation report given as the option 'validation' (the
validation.json that Help ▸ Validate or `--validate OUT` writes) gives the checks passed per model; without it the
paragraph says how many checks are registered for the model and that the suite was not run for this kit.

Settings (project.analysis.studies['publication'], options override; DEFAULTS, each a DexVantage default): sections,
siunitx, widths ('single', 'double'), pdf_figures, bibliography ('bibtex' or 'biblatex'), title, and the export
options of the report (results, results_dir, studies), validation and figure_specs (extra core/figures specs, such
as a page's chart).

Known limits: the paragraph describes what the project and the results state and nothing else (no task
description, no claims about the data the rig will record); stream rates and sizes are the ones the bandwidth model
selects (the rig's links, else its labelled default selection); a field of view is the maker's for the native image
(or the stated mode of that aspect), not re-derived for a cropped stream; a sensor whose device the library does not
hold is described by DexVantage's default camera; only the per-item chart each result has (report.figure) and the
Sobol indices of an experiment are drawn as figures, plus any spec passed in; characters outside Latin-1 with no
TeX equivalent in TEX_UNICODE are written as '?'; the LaTeX check is structural (braces, environments, \\input
targets), the kit is compiled only where a LaTeX engine is installed.
"""
from __future__ import annotations

import json
import math
import os
import re

from . import jobs
from .jobs import StudyError

SCHEMA = 'dexvantage.paper_kit/1'
STUDY_ID = 'publication'                       # its settings' key in project.analysis.studies
FOLDER = 'paper_kit'
SECTIONS = ('methods', 'sensors', 'results', 'scorecard', 'experiments', 'variants', 'statistics', 'cells', 'runs',
            'figures', 'appendix')
SECTION_TITLES = {'methods': 'Methods paragraph', 'sensors': 'Sensor table', 'results': 'Results table',
                  'scorecard': 'Requirements scorecard', 'experiments': 'Experiment results',
                  'variants': 'Variant comparison', 'statistics': 'Statistics (tests and intervals)',
                  'cells': 'Cell comparison', 'runs': 'Tracked runs', 'figures': 'Figures (PDF and SVG)',
                  'appendix': 'Reproducibility appendix'}
TABLE_PARTS = ('sensors', 'results', 'scorecard', 'experiments', 'variants', 'statistics', 'cells', 'runs')
RUNS_ROWS = 20                 # DexVantage default: the newest tracked runs listed in the runs table
WIDTHS = ('single', 'double')
DEFAULTS = {'sections': list(SECTIONS), 'siunitx': False, 'widths': list(WIDTHS), 'pdf_figures': True,
            'bibliography': 'bibtex', 'title': '',
            'decimals_m': 3,           # DexVantage default: metres to the millimetre
            'decimals_quat': 4}        # DexVantage default
SETTING_KEYS = ('sections', 'siunitx', 'widths', 'pdf_figures', 'bibliography', 'title')
META_TABLES = ('requirements', 'variants_compare', 'experiments', 'tolerance', 'summary', 'optimise',
               'design_agent', 'stats', 'cells',
               'tracking')        # results with a table of their own (or none) instead of rows of named figures
# the validation models (validation/suite.py MODELS) that check each study's model; None: no model of its own
STUDY_MODELS = {
    'summary': (), 'action_chunk_sync': ('timing',), 'audit': ('data',), 'bandwidth': ('bandwidth',),
    'brackets': ('brackets',), 'cables': ('cables',), 'canonical_view': ('vla',), 'coverage': ('occlusion', 'optics'),
    'depth_validity': ('depth',), 'dex_occlusion': ('dexhand',), 'experiments': ('experiments',),
    'grasp_visibility': ('grasp', 'kinematics'), 'groundtruth': ('groundtruth',), 'handeye_plan': ('handeye',),
    'handeye_solve': ('handeye',), 'lighting': ('lighting',), 'motion_blur': ('timing',), 'optimise': ('optimise',),
    'design_agent': ('optimise',), 'plan_vs_recording': ('data',), 'requirements': ('requirements',),
    'variants_compare': ('requirements',), 'rig': ('library',), 'silhouette_calib': (), 'teleop': ('teleop',),
    'time_sync': ('timing',), 'tolerance': ('tolerance',), 'vla_tokens': ('vla',), 'stats': ('stats',),
    'cells': ('cells',), 'tracking': ('tracking',), 'campaign': ('campaign',), 'sim2real': ('sim2real',)}
# metric -> (low path, high path, scale, level path, method) of the interval the study states for it
INTERVALS = {'grasp.contact_visibility': (('summary', 'ci_low'), ('summary', 'ci_high'), 100.0,
                                          ('summary', 'ci_level'), ('summary', 'ci_method'))}
# units siunitx is given as macros; any other unit is literal \unit{...} input or (count, tokens, ratio) plain text
SI_UNITS = {'%': r'\percent', '°': r'\degree', 'mm': r'\milli\metre', 'm': r'\metre', 'ms': r'\milli\second',
            's': r'\second', 'g': r'\gram', 'W': r'\watt', 'm/s': r'\metre\per\second',
            'mm/s': r'\milli\metre\per\second'}
LITERAL_UNITS = ('px', 'px/mm', 'px/s', 'Mbit/s', 'fps', '°/s')
TEXT_UNITS = ('count', 'tokens', 'ratio', 'USD', '')
TEX_SPECIAL = {'\\': r'\textbackslash{}', '{': r'\{', '}': r'\}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
               '_': r'\_', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}', '<': r'\textless{}',
               '>': r'\textgreater{}'}
TEX_UNICODE = {'×': r'$\times$', '°': r'$^\circ$', '±': r'$\pm$', 'σ': r'$\sigma$', 'µ': r'$\mu$', 'μ': r'$\mu$',
               'Δ': r'$\Delta$', 'ρ': r'$\rho$', '≥': r'$\geq$', '≤': r'$\leq$', '−': r'$-$', '–': '--',
               '—': '---', '’': "'", '‘': '`', '“': '``', '”': "''", '…': r'\ldots{}', '²': r'\textsuperscript{2}',
               '³': r'\textsuperscript{3}', '·': r'$\cdot$', '→': r'$\rightarrow$', '\u00a0': '~', '≈': r'$\approx$',
               'θ': r'$\theta$', 'π': r'$\pi$', 'ϕ': r'$\phi$', 'φ': r'$\phi$', 'λ': r'$\lambda$', '⁻': '-',
               '¹': r'\textsuperscript{1}', '•': r'$\bullet$', '✓': 'yes', '✗': 'no'}
DVNUM = r'\providecommand{\dvnum}[1]{\ensuremath{#1}}'


class PublicationError(StudyError):
    """The kit cannot be written as asked; the message says why."""


# ====================================================================== TeX text
def tex(text) -> str:
    """Plain text as TeX: the special characters escaped, common symbols as TeX, other characters beyond Latin-1
    written '?'."""
    out = []
    for c in str(text if text is not None else ''):
        if c in TEX_SPECIAL:
            out.append(TEX_SPECIAL[c])
        elif c in TEX_UNICODE:
            out.append(TEX_UNICODE[c])
        elif ord(c) > 0xFF:
            out.append('?')
        elif c in '\r\n\t':
            out.append(' ')
        else:
            out.append(c)
    return ''.join(out)


def _label(s) -> str:
    return re.sub(r'[^A-Za-z0-9:.-]+', '-', str(s)).strip('-') or 'x'


def number_text(v, decimals: int | None = None) -> str:
    """The number as written: decimals fixed digits after the point, else integers whole and others to 4 significant
    digits (exponent form 'e' beyond 1e-3 .. 1e6). '-0.000' is written '0.000'."""
    if isinstance(v, bool):
        v = int(v)
    if isinstance(v, int):
        return str(v)
    v = float(v)
    if decimals is not None:
        s = f'{v:.{int(decimals)}f}'
    elif v == int(v) and abs(v) < 1e15:
        s = str(int(v))
    elif abs(v) != 0 and (abs(v) < 1e-3 or abs(v) >= 1e6):
        s = f'{v:.3e}'
        m, e = s.split('e')
        s = f'{m}e{int(e)}'
    else:
        s = f'{v:.4g}'
        if 'e' in s:
            s = f'{v:.1f}'
    if re.fullmatch(r'-0(\.0*)?', s):
        s = s[1:]
    return s


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


class Writer:
    """Writes numbers, quantities and traced text, recording where each number comes from (module docstring)."""

    def __init__(self, siunitx: bool = False):
        self.si = bool(siunitx)
        self.trace: list = []
        self.cited: list = []

    # --------------------------------------------------------------- numbers
    def wrap(self, s: str) -> str:
        if self.si:
            return r'\num{' + s + '}'
        if 'e' in s:
            m, e = s.split('e')
            return r'\dvnum{' + m + r'\times10^{' + e + '}}'
        return r'\dvnum{' + s + '}'

    def n(self, value, path, decimals: int | None = None, scale: float = 1.0, kind: str = 'number') -> str:
        """A number read from path (value already read from it; scale applied) written and traced."""
        s = number_text(value * scale if scale != 1.0 and _is_num(value) else value, decimals)
        self.trace.append({'text': s, 'kind': kind, 'path': list(path), 'scale': scale})
        return self.wrap(s)

    def count(self, items, path) -> str:
        s = str(len(items))
        self.trace.append({'text': s, 'kind': 'count', 'path': list(path)})
        return self.wrap(s)

    def enabled(self, sensors, path) -> str:
        s = str(sum(1 for x in sensors if getattr(x, 'enabled', True)))
        self.trace.append({'text': s, 'kind': 'enabled', 'path': list(path)})
        return self.wrap(s)

    def metric(self, value, sid: str, mid: str, decimals=None) -> str:
        s = number_text(value, decimals)
        self.trace.append({'text': s, 'kind': 'metric', 'path': ['results', sid], 'metric': mid})
        return self.wrap(s)

    def t(self, text, path) -> str:
        """Text read from path, escaped; traced when it holds a digit (a name such as "Camera 2", a version)."""
        raw = str(text)
        if re.search(r'\d', raw):
            self.trace.append({'text': raw, 'kind': 'text', 'path': list(path)})
        return tex(raw)

    def unit(self, u: str) -> str:
        u = str(u or '')
        if not u or u in TEXT_UNITS:
            return tex(u)
        if self.si and u in SI_UNITS:
            return r'\unit{' + SI_UNITS[u] + '}'
        if self.si and re.fullmatch(r'[A-Za-z/]+', u):
            return r'\unit{' + u + '}'
        return tex(u)

    def q(self, value, path, unit: str, decimals=None, scale: float = 1.0) -> str:
        """A quantity: the number and its unit (\\qty with siunitx)."""
        s = number_text(value * scale if scale != 1.0 and _is_num(value) else value, decimals)
        self.trace.append({'text': s, 'kind': 'number', 'path': list(path), 'scale': scale})
        if self.si and unit in SI_UNITS:
            return r'\qty{' + s + '}{' + SI_UNITS[unit] + '}'
        if unit == '°':
            return self.wrap(s) + r'$^\circ$'
        if unit == '%':
            return self.wrap(s) + r'\,\%'
        return self.wrap(s) + r'~' + tex(unit)

    def cite(self, keys) -> str:
        ks = [k for k in dict.fromkeys(keys) if k]
        if not ks:
            return ''
        for k in ks:
            if k not in self.cited:
                self.cited.append(k)
        return r'~\cite{' + ','.join(ks) + '}'


# ====================================================================== sources and their check
def resolve(source: dict, path):
    """The value at path in the source data (KeyError/IndexError/TypeError when there is none)."""
    o = source
    for k in path:
        o = o[k]
    return o


def source_data(project, results: dict, validation: dict | None = None, scorecard: dict | None = None) -> dict:
    """Everything a number in the kit may come from (module docstring)."""
    from . import library as LB
    from ..validation import suite as VS
    pd = project.to_dict()
    pd.pop('log', None)
    ids = [s.model_id for v in project.variants for s in v.sensors if s.model_id]
    lib = {}
    for i in ids:
        e = LB.get(i)
        if e is not None:
            lib[i] = e.data
    failed = load_check_modules()
    reg: dict = {}
    for sp in VS.CHECKS:
        reg.setdefault(sp.model, []).append(sp.id)
    return {'project': pd, 'library': lib, 'results': results, 'environment': environment(),
            'validation': validation, 'registry': reg, 'registry_failed': failed, 'scorecard': scorecard,
            'defaults': {'resolution': list(LB.DEFAULT_RESOLUTION), 'hfov_deg': LB.DEFAULT_HFOV_DEG,
                         'vfov_deg': LB.DEFAULT_VFOV_DEG}}


def load_check_modules() -> list:
    """Import every validation check module (validation/suite.py load_checks, one module at a time); the names of
    those that fail to import (their checks are then not counted, and the paragraph says the registry is
    incomplete)."""
    import importlib
    import pkgutil

    from ..validation import suite as VS
    failed = []
    for m in pkgutil.iter_modules([os.path.dirname(os.path.abspath(VS.__file__))]):
        if m.name.startswith('checks_'):
            try:
                importlib.import_module(f'{VS.__package__}.{m.name}')
            except Exception:                           # noqa: BLE001 — named in the paragraph, never a traceback
                failed.append(m.name)
    return failed


def environment() -> dict:
    """core.provenance.environment() when Qt is already loaded (the window, an export); otherwise the same fields
    with the Qt and PyQt versions read from the installed packages' metadata, so the headless API never imports Qt
    (provenance.environment imports PyQt6.QtCore for them)."""
    import sys

    from . import provenance as PV
    if 'PyQt6.QtCore' in sys.modules:
        return PV.environment()
    sys.modules['PyQt6.QtCore'] = None                     # provenance's "from PyQt6.QtCore import" fails: no Qt
    try:
        env = PV.environment()
    finally:
        if sys.modules.get('PyQt6.QtCore', 0) is None:
            del sys.modules['PyQt6.QtCore']
    import importlib.metadata as md
    for key, pkg in (('qt', 'PyQt6-Qt6'), ('pyqt', 'PyQt6')):
        if not env['runtime'].get(key):
            try:
                env['runtime'][key] = md.version(pkg)
            except md.PackageNotFoundError:
                pass
    return env


_STRIP = re.compile(r'\\(?:cite[tp]?|ref|label|eqref|input|include|includegraphics|url|href|bibliography'
                    r'|bibliographystyle|addbibresource)\*?(?:\[[^\]]*\])*\{[^{}]*\}')
_NUM = re.compile(r'(?<![A-Za-z0-9_.\\])-?\d+(?:\.\d+)*(?![A-Za-z0-9_]|\.\d)')


def strip_comments(text: str) -> str:
    return '\n'.join(re.sub(r'(?<!\\)%.*$', '', line) for line in str(text).split('\n'))


def numbers_in(text: str) -> list:
    """Every number written in a TeX text (comments, citation keys, labels and file names left out)."""
    t = re.sub(r'^\\(?:provide|new|renew)command.*$', '', strip_comments(text), flags=re.M)
    return _NUM.findall(_STRIP.sub(' ', t))


def _close(token: str, v: float) -> bool:
    t = float(token)
    d = len(token.split('.', 1)[1]) if '.' in token and token.count('.') == 1 else 0
    return abs(v - t) <= 0.5 * 10 ** (-d) * (1 + 1e-9) + 1e-12


def check_entry(source: dict, e: dict) -> str:
    """'' when the trace entry's text is what its source states, else why not."""
    from . import metrics as M
    try:
        if e['kind'] == 'metric':
            M.load_metrics()
            r = M.read(M.get(e['metric']), resolve(source, e['path']))
            v = r['value']
            return '' if v is not None and _close(e['text'], v) else f'{e["metric"]} reads {v}, not {e["text"]}'
        v = resolve(source, e['path'])
    except (KeyError, IndexError, TypeError, StudyError):
        return f'{e["text"]}: nothing at {"/".join(map(str, e["path"]))}'
    if e['kind'] == 'enabled':
        return '' if isinstance(v, list) and sum(1 for x in v if x.get('enabled', True)) == int(e['text']) else \
            f'{e["text"]} is not the number of enabled items of {"/".join(map(str, e["path"]))}'
    if e['kind'] == 'count':
        return '' if isinstance(v, (list, dict)) and len(v) == int(e['text']) else \
            f'{e["text"]} is not the length of {"/".join(map(str, e["path"]))}'
    if e['kind'] == 'text':
        return '' if str(v) == e['text'] or e['text'] in str(v) else f'"{e["text"]}" is not in the source text'
    if not _is_num(v):
        return f'{e["text"]}: {"/".join(map(str, e["path"]))} is not a number'
    return '' if _close(e['text'], float(v) * float(e.get('scale', 1.0))) else \
        f'{e["text"]} is not {v} rounded'


def verify_numbers(text: str, trace: list, source: dict) -> dict:
    """Every number in text matched to a trace entry, and every entry checked against the source data."""
    problems = [p for p in (check_entry(source, e) for e in trace) if p]
    pool = [e['text'] for e in trace if e['kind'] != 'text']
    texts = [e['text'] for e in trace if e['kind'] == 'text']
    toks = numbers_in(text)
    untraced = []
    for t in toks:
        if t in pool:
            pool.remove(t)
            continue
        if any(t in numbers_in(tex(x)) or t in numbers_in(x) for x in texts):
            continue
        untraced.append(t)
    problems += [f'{t} has no source' for t in untraced]
    return {'numbers': len(toks), 'traced': len(toks) - len(untraced), 'problems': problems}


# ====================================================================== structural checks
_CITE = re.compile(r'\\(?:no)?cite[tp]?\*?(?:\[[^\]]*\])*\{([^{}]*)\}')


def cited_keys(text: str) -> list:
    out = []
    for m in _CITE.finditer(strip_comments(text)):
        for k in m.group(1).split(','):
            k = k.strip()
            if k and k not in out:
                out.append(k)
    return out


def bib_keys(bib: str) -> list:
    return re.findall(r'^@[A-Za-z]+\{([^,\s]+),', bib, re.M)


def latex_problems(text: str) -> list:
    """Unbalanced braces and mismatched \\begin/\\end (verbatim content skipped), in plain sentences."""
    probs = []
    stack: list = []
    depth = 0
    in_verb = None
    for ln, line in enumerate(str(text).split('\n'), 1):
        if in_verb:
            if re.search(r'\\end\{' + re.escape(in_verb) + r'\}', line):
                in_verb = None
                stack.pop()
            continue
        line = strip_comments(line)
        i = 0
        while i < len(line):
            c = line[i]
            if c == '\\':
                m = re.match(r'\\(begin|end)\{([^{}]*)\}', line[i:])
                if m:
                    name = m.group(2)
                    if m.group(1) == 'begin':
                        stack.append((name, ln))
                        if name in ('verbatim', 'verbatim*', 'lstlisting'):
                            in_verb = name
                            break
                    elif not stack:
                        probs.append(f'line {ln}: \\end{{{name}}} without a \\begin')
                    elif stack[-1][0] != name:
                        probs.append(f'line {ln}: \\begin{{{stack[-1][0]}}} (line {stack[-1][1]}) ended by '
                                     f'\\end{{{name}}}')
                        stack.pop()
                    else:
                        stack.pop()
                    i += len(m.group(0))
                    continue
                i += 2
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth < 0:
                    probs.append(f'line {ln}: a closing brace without an opening one')
                    depth = 0
            i += 1
    if depth:
        probs.append(f'{depth} brace(s) left open')
    for name, ln in stack:
        probs.append(f'\\begin{{{name}}} (line {ln}) is never ended')
    return probs


# ====================================================================== citations
class CiteMap:
    """The BibTeX entries of core/cite.py (references 'all') and which key cites what."""

    def __init__(self, project):
        from . import cite as C
        from . import library as LB
        self.entries = C.entries(project, 'all')
        self.by_key = {e.key: e for e in self.entries}
        self.self_key = next((e.key for e in self.entries if e.group == 'DexVantage'), 'dexvantage')
        self._by_ref: dict = {}
        self._refs: dict = {}
        self._robots: dict = {}
        for e in self.entries:
            if e.group.startswith('Library sources'):
                for f in ('url', 'howpublished'):
                    if e.fields.get(f):
                        self._by_ref.setdefault(e.fields[f], e.key)
            elif e.group.startswith('References of the validation'):
                m = re.search(r'\(reference (.+?)\)', e.fields.get('note', '').replace('\\_', '_'))
                if m:
                    self._refs[m.group(1)] = e.key
        self.library = {}
        for mid in {s.model_id for v in project.variants for s in v.sensors if s.model_id}:
            ent = LB.get(mid)
            if ent is None:
                continue
            ks = []
            for s in ent.sources:
                if not isinstance(s, dict):
                    continue
                k = self._by_ref.get(str(s.get('url') or '')) or self._by_ref.get(C.escape(str(s.get('document') or
                                                                                                   '')))
                if k and k not in ks:
                    ks.append(k)
            self.library[mid] = ks
        robot_entries = [e for e in self.entries if e.group == 'Robot models']
        soft = next((e.key for e in robot_entries if e.type == 'software'), None)
        for r in project.robots:
            ks = []
            models = C.menagerie_models(r)
            for folder, _c in models:
                k = next((e.key for e in robot_entries if str(e.fields.get('url', '')).endswith('/' + folder)), None)
                if k and k not in ks:
                    ks.append(k)
            if models and soft:
                ks.insert(0, soft)
            if not models:
                k = next((e.key for e in robot_entries if e.fields.get('title') == '{' + C.escape(r.name) + '}'), None)
                if k:
                    ks.append(k)
            self._robots[r.uid] = ks

    def robot(self, uid) -> list:
        return list(self._robots.get(uid, []))

    def ref(self, key) -> str | None:
        return self._refs.get(key)

    def bib(self, project, keys) -> str:
        from . import cite as C
        used = [e for e in self.entries if e.key in set(keys)]
        return C.bibtex(project, es=used)


# ====================================================================== settings
def settings_of(project, options: dict | None = None) -> dict:
    """DEFAULTS, then the project's stored kit settings, then options; checked (PublicationError on a bad value)."""
    st = dict(DEFAULTS)
    stored = (project.analysis.studies.get(STUDY_ID) or {}) if project is not None else {}
    for k in SETTING_KEYS:
        if k in stored:
            st[k] = stored[k]
    for k, v in (options or {}).items():
        st[k] = v
    secs = st.get('sections')
    if isinstance(secs, str):
        secs = [s.strip() for s in secs.split(',') if s.strip()]
    if not isinstance(secs, (list, tuple)) or any(s not in SECTIONS for s in secs):
        raise PublicationError(f'The kit\'s sections are some of {", ".join(SECTIONS)}.')
    st['sections'] = [s for s in SECTIONS if s in secs]
    w = st.get('widths')
    if isinstance(w, str):
        w = [x.strip() for x in w.split(',') if x.strip()]
    if not isinstance(w, (list, tuple)) or not w or any(x not in WIDTHS for x in w):
        raise PublicationError('The figure widths are "single" (one column), "double" (two columns) or both.')
    st['widths'] = [x for x in WIDTHS if x in w]
    if st.get('bibliography') not in ('bibtex', 'biblatex'):
        raise PublicationError('The bibliography is "bibtex" or "biblatex".')
    st['siunitx'] = bool(st.get('siunitx'))
    st['pdf_figures'] = bool(st.get('pdf_figures', True))
    return st


def stored_settings(st: dict) -> dict:
    """The settings to keep in the project (project.analysis.studies['publication'])."""
    return {k: st[k] for k in SETTING_KEYS if k in st}


# ====================================================================== the methods paragraph
def _sensor_streams(project, sensor, entry):
    """The streams the bandwidth model takes for this sensor (the rig's link, else its default selection)."""
    from .studies import bandwidth as BW
    link = next((k for k in (project.rig or {}).get('links') or [] if k.get('sensor_uid') == sensor.uid), None)
    try:
        return BW.sensor_streams(sensor, entry, (link or {}).get('stream_ids') or [], BW.settings_of(project))
    except (StudyError, KeyError, TypeError, ValueError):
        return {'streams': [], 'default': True}


def _stream_path(entry, row):
    """(stream index, fps index) of a selected stream in the entry's 'streams'."""
    from .library import _norm
    for i, s in enumerate(entry.data.get('streams') or []):
        if (_norm(s.get('imager')) == _norm(row['imager']) and int(s.get('width', 0)) == row['width']
                and int(s.get('height', 0)) == row['height'] and (s.get('format') or None) == (row['format'] or None)):
            for j, f in enumerate(s.get('fps') or []):
                if _is_num(f) and abs(float(f) - float(row['fps'])) < 1e-9:
                    return i, j
            return i, None
    return None, None


def _imager_for(entry, name):
    for i, im in enumerate(entry.imagers):
        if im is entry.imager(name):
            return i, im
    return None, None


def stated_fov(entry, row) -> dict | None:
    """The FOV the maker states for a selected stream's imager: the stated mode of the stream's aspect when the
    stream is not the native image, else the imager's own (for its native image). {'h', 'v', 'h_path', 'v_path'
    (relative to the library entry), 'mode': True for a stated mode, 'native'}; None when no horizontal FOV is
    stated."""
    from .library import _aspect_eq
    k, im = _imager_for(entry, row['imager'])
    if im is None:
        return None
    base = ['imagers', k]
    w, h = row['width'], row['height']
    native = im.get('resolution')
    for j, fm in enumerate(im.get('fov_modes') or []):
        if fm.get('hfov_deg') is not None and native and tuple(native) != (w, h) and _aspect_eq((w, h), fm['aspect']):
            return {'h': fm['hfov_deg'], 'v': fm.get('vfov_deg'), 'h_path': base + ['fov_modes', j, 'hfov_deg'],
                    'v_path': base + ['fov_modes', j, 'vfov_deg'], 'mode': True, 'native': native, 'base': base}
    if im.get('hfov_deg') is None:
        return None
    return {'h': im['hfov_deg'], 'v': im.get('vfov_deg'), 'h_path': base + ['hfov_deg'], 'v_path': base + ['vfov_deg'],
            'mode': False, 'native': native, 'base': base}


def _fov_text(W: Writer, entry, row, mid, which: str = '') -> str:
    """The stated FOV of the stream's imager (stated_fov), with where it is stated; which (TeX) names the stream it
    belongs to when the sentence lists several."""
    f = stated_fov(entry, row)
    if f is None:
        return 'a field of view the maker does not state' if _imager_for(entry, row['imager'])[1] else ''
    lp = ['library', mid]
    v = r' $\times$ ' + W.q(f['v'], lp + f['v_path'], '°') if _is_num(f['v']) else ''
    hv = f'{W.q(f["h"], lp + f["h_path"], "°")}{v}' + (f' for its {which} stream' if which else '')
    if f['mode']:
        return f'a field of view of {hv} (horizontal by vertical, as the maker states for streams of this aspect)'
    where = ''
    if f['native']:
        b = lp + f['base']
        where = (f' for the native {W.n(f["native"][0], b + ["resolution", 0])}~$\\times$~'
                 f'{W.n(f["native"][1], b + ["resolution", 1])}~px image')
    return f'a field of view of {hv} (horizontal by vertical, as the maker states{where})'


def _pose_text(W: Writer, pose: dict, path: list, st: dict) -> str:
    dm, dq = int(st['decimals_m']), int(st['decimals_quat'])
    pos = ', '.join(W.n(pose['position'][i], path + ['position', i], dm) for i in range(3))
    qu = ', '.join(W.n(pose['quaternion'][i], path + ['quaternion', i], dq) for i in range(4))
    unit = r'~\unit{\metre}' if W.si else '~m'
    return f'position ({pos}){unit} and orientation (x, y, z, w) = ({qu})'


def _sensor_sentence(project, W: Writer, cm: CiteMap, vi: int, si: int, s, st: dict) -> str:
    from . import library as LB
    path = ['project', 'variants', vi, 'sensors', si]
    name = W.t(s.name, path + ['name'])
    entry = LB.get(s.model_id) if s.model_id else None
    parts = []
    if entry is None:
        d = ['defaults']
        why = f' (its device "{tex(s.model_id)}" is not in the library)' if s.model_id else ''
        parts.append(f'{name} is a {tex(s.kind)} with no device chosen{why}: the studies use DexVantage\'s '
                     f'default camera model of {W.n(LB.DEFAULT_RESOLUTION[0], d + ["resolution", 0])}~$\\times$~'
                     f'{W.n(LB.DEFAULT_RESOLUTION[1], d + ["resolution", 1])}~px with a '
                     f'{W.q(LB.DEFAULT_HFOV_DEG, d + ["hfov_deg"], "°")} $\\times$ '
                     f'{W.q(LB.DEFAULT_VFOV_DEG, d + ["vfov_deg"], "°")} field of view (DexVantage defaults, not a '
                     f'device\'s figures)')
    else:
        mid = s.model_id
        lp = ['library', mid]
        dev = tex(entry.title)
        for part in ('maker', 'name'):                # digits in the device's name ("D405") are traced as text
            W.t(entry.data.get(part) or '', lp + [part])
        what = {'camera': 'camera', 'depth': 'depth camera', 'tactile': 'tactile sensor'}.get(s.kind, tex(s.kind))
        d = entry.data.get('depth') or {}
        extra = ''
        if s.kind == 'depth' and d:
            tech = re.sub(r'\bir\b', 'IR', str(d.get('technology') or '').replace('_', ' '))
            if tech:
                what = f'{tex(tech)} depth camera'
            if _is_num(d.get('baseline_m')):
                extra = f' with a stereo baseline of {W.q(d["baseline_m"], lp + ["depth", "baseline_m"], "m")}'
        art = 'an' if re.match(r'[AEIOU]', entry.title) else 'a'
        parts.append(f'{name} is {art} {dev}{W.cite(cm.library.get(mid, []))} {what}{extra}')
        if s.kind == 'tactile':
            tc = entry.data.get('tactile') or {}
            bits = []
            if tc.get('sensing'):
                bits.append(f'{tex(tc["sensing"])} sensing')
            pad = tc.get('pad_mm')
            if isinstance(pad, list) and len(pad) == 2 and all(_is_num(x) for x in pad):
                u = r'~\unit{\milli\metre}' if W.si else '~mm'
                bits.append(f'a {W.n(pad[0], lp + ["tactile", "pad_mm", 0])}~$\\times$~'
                            f'{W.n(pad[1], lp + ["tactile", "pad_mm", 1])}{u} pad')
            if _is_num(tc.get('fps')):
                bits.append(f'{W.n(tc["fps"], lp + ["tactile", "fps"])}~fps')
            if _is_num(tc.get('taxels')):
                bits.append(f'{W.n(tc["taxels"], lp + ["tactile", "taxels"])} taxels')
            if bits:
                parts.append('(' + ', '.join(bits) + ')')
        ss = _sensor_streams(project, s, entry)
        rows = [r for r in ss.get('streams') or [] if not r.get('extra')]
        if rows:
            descr = []
            for r in rows:
                i, j = _stream_path(entry, r)
                if i is None:
                    continue
                sp = lp + ['streams', i]
                fps = W.n(r['fps'], sp + ['fps', j]) if j is not None else W.n(r['fps'], sp + ['fps', 0])
                descr.append(f'its {W.t(r["imager"], sp + ["imager"])} stream at {W.n(r["width"], sp + ["width"])}'
                             f'~$\\times$~{W.n(r["height"], sp + ["height"])}~px and {fps}~fps'
                             + (f' ({tex(r["format"])})' if r.get('format') else ''))
            if descr:
                how = 'the stream DexVantage selects by default when the rig names none' if ss.get('default') else \
                    'as the rig\'s link names'
                parts.append(f'; the studies take {", ".join(descr)} ({how})')
            i0 = _stream_path(entry, rows[0])[0]
            which = W.t(rows[0]['imager'], lp + ['streams', i0, 'imager']) if len(descr) > 1 and i0 is not None \
                else ''                                # the FOV is the first stream's imager's, not the last one's
            fov = _fov_text(W, entry, rows[0], mid, which)
            if fov:
                parts.append(f', with {fov}')
        elif s.kind != 'tactile':
            parts.append('; the maker lists no stream mode for it')
    if s.calibration:
        parts.append('; its intrinsics come from the unit\'s own calibration')
    m = s.mount
    mp = path + ['mount']
    pose = s.to_dict()['mount']['pose']
    rob = project.robot(m.robot_uid) if m.robot_uid else None
    if m.on_link and rob is not None:
        ri = project.robots.index(rob)
        where = (f'mounted on link \\texttt{{{W.t(m.parent_frame, mp + ["parent_frame"])}}} of '
                 f'{W.t(rob.name, ["project", "robots", ri, "name"])} at {_pose_text(W, pose, mp + ["pose"], st)} in '
                 f'that link\'s frame')
    else:
        frame = W.t(m.parent_frame, mp + ['parent_frame'])
        where = (f'mounted on the static frame \\texttt{{{frame}}} at {_pose_text(W, pose, mp + ["pose"], st)} in '
                 f'the world frame')
    return ''.join(parts) + f'. It is {where}.'


def _robot_sentence(project, W: Writer, cm: CiteMap) -> str:
    from . import cite as C
    if not project.robots:
        return 'The project holds no robot model.'
    out = []
    n = len(project.robots)
    if n > 1:
        out.append(f'The cell holds {W.count(project.robots, ["project", "robots"])} robots.')
    for i, r in enumerate(project.robots):
        rp = ['project', 'robots', i]
        name = W.t(r.name, rp + ['name'])
        models = C.menagerie_models(r)
        lic = W.t(r.source.license, rp + ['source', 'license']) if r.source.license else ''
        if models:
            folders = ', '.join(f'\\texttt{{{W.t(f, rp + ["source", "note"])}}}' for f, _c in models)
            commit = models[0][1]
            where = 'note' if commit and commit in r.source.note else 'license'
            ctext = (f' at commit \\texttt{{{W.t(commit, rp + ["source", where])}}}' if commit and
                     commit in (r.source.note + r.source.license) else '')
            src = f'the MuJoCo Menagerie model{"s" if len(models) > 1 else ""} {folders}{ctext}'
        elif r.source.file:
            fn = W.t(r.source.file, rp + ['source', 'file'])
            src = f'its {tex(r.source.kind.upper() or "model")} file \\texttt{{{fn}}}'
        else:
            src = 'a model whose source the project does not record'
        role = 'the primary robot, whose base is the world frame (x forward, y left, z up)' if r is \
            project.primary_robot else 'a further robot'
        s = f'{name} is {role}, modelled from {src}{W.cite(cm.robot(r.uid))}'
        if lic:
            s += f' (licence: {lic})'
        if r is not project.primary_robot:
            s += (f'; its base is at {_pose_text(W, r.to_dict()["base"], rp + ["base"], DEFAULTS)} in the world '
                  f'frame')
        out.append(s + '.')
    return ' '.join(out)


def _software_sentence(W: Writer, cm: CiteMap, env: dict) -> str:
    dv = env.get('dexvantage') or {}
    rt = env.get('runtime') or {}
    ep = ['environment']
    s = (f'We planned the sensing rig with DexVantage{W.cite([cm.self_key])} version '
         f'\\texttt{{{W.t(dv.get("version") or "unknown", ep + ["dexvantage", "version"])}}}')
    git = dv.get('git') or {}
    if git.get('commit'):
        s += f' (commit \\texttt{{{W.t(str(git["commit"])[:12], ep + ["dexvantage", "git", "commit"])}}}'
        s += ', with local changes)' if git.get('dirty') else ')'
    if rt.get('mujoco'):
        s += (f', which loads the robot models with MuJoCo{W.cite([cm.ref("mujoco_todorov2012")])} version '
              f'{W.t(rt["mujoco"], ep + ["runtime", "mujoco"])}')
    py = rt.get('python')
    if py:
        s += f' under Python {W.t(py, ep + ["runtime", "python"])}'
    return s + '.'


def _settings_text(W: Writer, project, sid: str) -> str:
    st = project.analysis.studies.get(sid) or {}
    bits = []
    for k, v in st.items():
        p = ['project', 'analysis', 'studies', sid, k]
        if isinstance(v, bool):
            bits.append(f'{tex(k.replace("_", " "))} {"on" if v else "off"}')
        elif _is_num(v):
            bits.append(f'{tex(k.replace("_", " "))} {W.n(v, p)}')
        elif isinstance(v, str) and v and len(v) < 60 and not os.path.isabs(v):
            bits.append(f'{tex(k.replace("_", " "))} ``{W.t(v, p)}\'\'')
        if len(bits) >= 6:
            break
    return ', '.join(bits)


def _validation_text(W: Writer, cm: CiteMap, sid: str, source: dict) -> str:
    from ..validation import suite as VS
    models = STUDY_MODELS.get(sid)
    if models is None:
        return 'DexVantage registers no validation model for it'
    if not models:
        return 'it has no validation model of its own'
    titles = {m[0]: m[1] for m in VS.MODELS}
    val = source.get('validation')
    if source.get('registry_failed') and not isinstance(val, dict):
        return ('its validation checks could not be listed (the check module'
                f'{"s" if len(source["registry_failed"]) > 1 else ""} '
                f'{", ".join(tex(x) for x in source["registry_failed"])} failed to load)')
    out, refs = [], []
    for m in models:
        ids = source['registry'].get(m) or []
        title = tex(titles.get(m, m))
        row = None
        if isinstance(val, dict):
            mi = next((i for i, x in enumerate(val.get('models') or []) if x.get('id') == m), None)
            if mi is not None:
                row = (mi, val['models'][mi])
        if row is not None:
            mi, mr = row
            npass = W.n(mr.get('passed', 0), ['validation', 'models', mi, 'passed'])
            out.append(f'the model ``{title}\'\' passed {npass} '
                       f'of {W.count(mr.get("checks") or [], ["validation", "models", mi, "checks"])} checks')
        else:
            out.append(f'the model ``{title}\'\' has {W.count(ids, ["registry", m])} registered checks')
        for sp in VS.CHECKS:
            if sp.model == m:
                for r in sp.refs:
                    k = cm.ref(r)
                    if k and k not in refs and len(refs) < 2:
                        refs.append(k)
    return '; '.join(out) + W.cite(refs)


def methods(project, results: dict, source: dict, cm: CiteMap, W: Writer, st: dict, scorecard=None,
            refused: dict | None = None) -> str:
    """The methods paragraph (TeX) from the project and the results; numbers traced in W.trace."""
    paras = [_software_sentence(W, cm, source['environment']), _robot_sentence(project, W, cm)]
    v = project.variant
    vi = project.variants.index(v)
    on = [(i, s) for i, s in enumerate(v.sensors) if s.enabled]
    vp = ['project', 'variants', vi]
    head = f'The rig ``{W.t(v.name, vp + ["name"])}\'\''
    if len(project.variants) > 1:
        head += f' (one of {W.count(project.variants, ["project", "variants"])} rig variants)'
    if on:
        kinds = sorted({s.kind for _i, s in on})
        if len(on) == len(v.sensors):
            n_txt = f'{W.count(v.sensors, vp + ["sensors"])} {"sensor" if len(on) == 1 else "sensors"}'
        else:
            n_txt = f'{W.enabled(v.sensors, vp + ["sensors"])} enabled {"sensor" if len(on) == 1 else "sensors"}'
        head += (f' holds {n_txt} ({", ".join(tex(k) for k in kinds)}); positions are in metres and orientations '
                 'are unit quaternions, each in the frame of its mount.')
    else:
        head += ' holds no enabled sensor.'
    sens = [head] + [_sensor_sentence(project, W, cm, vi, i, s, st) for i, s in on]
    paras.append(' '.join(sens))
    a = project.analysis
    study_bits = []
    voxel_studies = [sid for sid in results if sid in ('coverage', 'depth_validity', 'teleop', 'lighting')]
    if voxel_studies:
        u = r'~\unit{\milli\metre}' if W.si else '~mm'
        study_bits.append(f'The volumetric studies sampled the task volumes on a voxel grid of '
                          f'{W.n(a.voxel_mm, ["project", "analysis", "voxel_mm"])}{u} '
                          f'({W.n(a.refine_mm, ["project", "analysis", "refine_mm"])}{u} inside refinement regions).')
    for sid, doc in results.items():
        if sid == 'summary':
            continue
        info = jobs.STUDIES.get(sid)
        title = tex(info.title if info else sid)
        rp = ['results', sid]
        s = f'{title} (study \\texttt{{{tex(sid)}}})'
        if sid == 'tolerance' and _is_num(doc.get('n')):
            s += f' drew {W.n(doc["n"], rp + ["n"])} perturbed rigs'
            if _is_num(doc.get('seed')):
                s += f' (seed {W.n(doc["seed"], rp + ["seed"])}, {tex(doc.get("distribution") or "")} draws)'
        elif sid == 'experiments' and isinstance(doc.get('runs'), list):
            s += (f' ran {W.count(doc["runs"], rp + ["runs"])} runs of a {tex(doc.get("method_label") or "")} '
                  f'design')
        else:
            s += ' ran'
        sett = _settings_text(W, project, sid)
        s += f' with {"the settings " + sett + " and the" if sett else "the"} DexVantage defaults otherwise'
        if doc.get('cancelled'):
            s += ' (stopped early: a partial result)'
        s += f'; {_validation_text(W, cm, sid, source)}.'
        study_bits.append(s)
    if refused:
        study_bits.append('Studies that could not run on this project are listed in the appendix.')
    if study_bits:
        paras.append(' '.join(study_bits))
    tables = []
    if 'sensors' in st['sections'] and on:
        tables.append(r'Table~\ref{tab:sensors} lists the sensors')
    if 'results' in st['sections'] and any(s not in META_TABLES for s in results):
        tables.append(r'Table~\ref{tab:results} the figures each study computed')
    if scorecard and scorecard.get('items') and 'scorecard' in st['sections']:
        tables.append(r'Table~\ref{tab:scorecard} the rig against its '
                      + W.count(scorecard['items'], ['scorecard', 'items']) + ' requirements')
    sd = results.get('stats')
    if 'statistics' in st['sections'] and isinstance(sd, dict) and sd.get('comparisons'):
        tables.append(r'Table~\ref{tab:statistics} the statistical comparisons with their intervals and corrected '
                      'p-values')
    cd = results.get('cells')
    if 'cells' in st['sections'] and isinstance(cd, dict) and (cd.get('metrics') or {}).get('rows'):
        tables.append(r'Table~\ref{tab:cells} the ' + W.count(cd.get('cells') or [], ['results', 'cells', 'cells'])
                      + ' robot cells compared')
    td = results.get('tracking')
    if 'runs' in st['sections'] and isinstance(td, dict) and td.get('runs'):
        tables.append(r'Table~\ref{tab:runs} the runs the experiment tracker logged')
    if tables:
        paras.append('; '.join(tables) + '.')
    return '\n\n'.join(paras) + '\n'


# ====================================================================== tables
def _row(cells: list, refs: list) -> str:
    return ' & '.join(cells) + r' \\' + (' % dv-src: ' + json.dumps(refs, ensure_ascii=False) if refs else '')


def _table(caption: str, label: str, cols: str, head: list, rows: list, wide: bool = False, note: str = '') -> str:
    env = 'table*' if wide else 'table'
    lines = ['% Written by DexVantage (publication kit); booktabs rules, units in the headings.', DVNUM,
             f'\\begin{{{env}}}[t]', r'\centering', r'\small', f'\\caption{{{caption}}}', f'\\label{{{label}}}']
    if wide:
        lines.append(r'\resizebox{\textwidth}{!}{%')
    lines += [f'\\begin{{tabular}}{{{cols}}}', r'\toprule', ' & '.join(head) + r' \\', r'\midrule']
    lines += rows or [r'\multicolumn{' + str(len(head)) + r'}{l}{(none)} \\']
    lines += [r'\bottomrule', r'\end{tabular}']
    if wide:
        lines.append('}')
    if note:
        lines.append(r'\par\smallskip\footnotesize ' + note)
    lines.append(f'\\end{{{env}}}')
    return '\n'.join(lines) + '\n'


def _h(W: Writer, title: str, unit: str = '') -> str:
    return tex(title) + (f' ({W.unit(unit)})' if unit else '')


def sensors_table(project, W: Writer) -> str:
    from . import library as LB
    v = project.variant
    vi = project.variants.index(v)
    rows = []
    for si, s in enumerate(v.sensors):
        if not s.enabled:
            continue
        sp = ['path', 'project', 'variants', vi, 'sensors', si]
        e = LB.get(s.model_id) if s.model_id else None
        refs, cells = [], [tex(s.name), tex(e.title) if e else 'default camera (no device)', tex(s.kind)]
        res = fps = fov = '---'
        if e is not None:
            ss = _sensor_streams(project, s, e)
            r = next((x for x in ss.get('streams') or [] if not x.get('extra')), None)
            if r is not None:
                i, j = _stream_path(e, r)
                if i is not None:
                    lp = ['path', 'library', s.model_id, 'streams', i]
                    res = f'{W.wrap(number_text(r["width"]))}$\\times${W.wrap(number_text(r["height"]))}'
                    refs += [lp + ['width'], lp + ['height']]
                    fps = W.wrap(number_text(r['fps']))
                    refs.append(lp + ['fps', j if j is not None else 0])
                f = stated_fov(e, r)
                if f is not None and _is_num(f['h']):
                    lp = ['path', 'library', s.model_id]
                    fov = W.wrap(number_text(f['h']))
                    refs.append(lp + f['h_path'])
                    if _is_num(f['v']):
                        fov += r'$\times$' + W.wrap(number_text(f['v']))
                        refs.append(lp + f['v_path'])
        else:
            dp = ['path', 'defaults']
            res = (f'{W.wrap(number_text(LB.DEFAULT_RESOLUTION[0]))}$\\times$'
                   f'{W.wrap(number_text(LB.DEFAULT_RESOLUTION[1]))}')
            fov = f'{W.wrap(number_text(LB.DEFAULT_HFOV_DEG))}$\\times${W.wrap(number_text(LB.DEFAULT_VFOV_DEG))}'
            refs += [dp + ['resolution', 0], dp + ['resolution', 1], dp + ['hfov_deg'], dp + ['vfov_deg']]
        rob = project.robot(s.mount.robot_uid) if s.mount.on_link else None
        mount = (tex(s.mount.parent_frame) + (f' ({tex(rob.name)})' if rob else ''))
        pos = s.to_dict()['mount']['pose']['position']
        xyz = ', '.join(W.wrap(number_text(pos[i], DEFAULTS['decimals_m'])) for i in range(3))
        refs += [sp + ['mount', 'pose', 'position', i] for i in range(3)]
        cells += [res, fps, fov, mount, xyz]
        # the order of refs must follow the cells: stream (w, h), fps, fov (h, v), position (x, y, z)
        rows.append(_row(cells, refs))
    head = ['Sensor', 'Device', 'Kind', _h(W, 'Stream', 'px'), _h(W, 'Rate', 'fps'), _h(W, 'FOV h × v', '°'),
            'Mount', _h(W, 'Position x, y, z', 'm')]
    return _table(f'Sensors of the rig ``{tex(v.name)}\'\' (devices and figures from the cited data sheets; '
                  'position in the mount\'s frame).', 'tab:sensors', 'lllrrrll', head, rows, wide=True)


def interval(mid: str, doc: dict):
    """(low, high, scale, level, method, low path, high path) of the interval a result states for a metric, or
    None."""
    spec = INTERVALS.get(mid)
    if spec is None:
        return None
    try:
        lo, hi = resolve(doc, spec[0]), resolve(doc, spec[1])
    except (KeyError, IndexError, TypeError):
        return None
    if not (_is_num(lo) and _is_num(hi)):
        return None
    try:
        level = resolve(doc, spec[3])
    except (KeyError, IndexError, TypeError):
        level = None
    try:
        method = resolve(doc, spec[4])
    except (KeyError, IndexError, TypeError):
        method = ''
    return lo, hi, spec[2], level, method, list(spec[0]), list(spec[1])


def results_table(results: dict, W: Writer) -> str:
    from .exporters import report as R
    rows = []
    for sid, doc in results.items():
        if sid in META_TABLES and sid != 'tolerance':
            continue
        info = jobs.STUDIES.get(sid)
        title = tex(info.title if info else sid)
        if sid == 'tolerance':
            for mid, m in (doc.get('metrics') or {}).items():
                mp = ['path', 'results', 'tolerance', 'metrics', mid]
                cells, refs = [title, tex(m.get('title') or mid) + ' (nominal; draws)'], []
                if _is_num(m.get('nominal')):
                    cells.append(W.wrap(number_text(m['nominal'])))
                    refs.append(mp + ['nominal'])
                else:
                    cells.append('not stated')
                cells.append(W.unit(m.get('unit') or ''))
                if _is_num(m.get('p5')) and _is_num(m.get('p95')):
                    cells.append(f'[{W.wrap(number_text(m["p5"]))}, {W.wrap(number_text(m["p95"]))}]')
                    refs += [mp + ['p5'], mp + ['p95']]
                    cells.append(f'P5--P95 of {W.wrap(number_text(m.get("valid", 0)))} draws')
                    refs.append(mp + ['valid'])
                else:
                    cells += ['---', '']
                rows.append(_row(cells, refs))
            continue
        for m in R.metric_rows(sid, doc):
            refs = []
            if m['value'] is None:
                val = 'not stated'
            else:
                val = W.wrap(number_text(m['value']))
                refs.append(['metric', sid, m['id']])
                if m['text'].startswith('≥'):
                    val = r'$\geq$ ' + val
                elif m['text'].startswith('≤'):
                    val = r'$\leq$ ' + val
            ci, kind = '---', ''
            iv = interval(m['id'], doc) if m['value'] is not None else None
            if iv is not None:
                lo, hi, sc, level, method, lp, hp = iv
                ci = f'[{W.wrap(number_text(lo * sc))}, {W.wrap(number_text(hi * sc))}]'
                refs += [['scaled', sc, 'results', sid] + lp, ['scaled', sc, 'results', sid] + hp]
                kind = (f'{W.wrap(number_text(level * 100))}\\,\\% ' if _is_num(level) else '') + tex(method)
                if _is_num(level):
                    refs.append(['scaled', 100.0, 'results', sid] + list(INTERVALS[m['id']][3]))
            rows.append(_row([title, tex(m['title']), val, W.unit(m['unit']), ci, kind], refs))
    head = ['Study', 'Figure', 'Value', 'Unit', 'Interval', 'Interval kind']
    return _table('Figures the studies computed (DexVantage\'s named metrics, as each study states them; '
                  '``not stated\'\' where a result does not state the figure).', 'tab:results', 'llrlll', head, rows,
                  wide=True)


def scorecard_table(card: dict | None, W: Writer, missing: list | None = None) -> str:
    if not card or not card.get('items'):
        why = ('the scorecard needs the results of ' + ', '.join(tex(m) for m in missing) + ' (run them first)'
               if missing else 'the project has no requirements')
        return _table('Requirements scorecard: ' + why + '.', 'tab:scorecard', 'l', ['Requirement'], [])
    rows = []
    for k, it in enumerate(card['items']):
        ip = ['path', 'scorecard', 'items', k]
        refs = []
        if it.get('value') is None:
            val = '---'
        else:
            val = W.wrap(number_text(it['value']))
            refs.append(ip + ['value'])
            if it.get('bound') == 'at_least':
                val = r'$\geq$ ' + val
            elif it.get('bound') == 'at_most':
                val = r'$\leq$ ' + val
        thr = tex(it.get('op_text') or it.get('op') or '') + ' ' + W.wrap(number_text(it['threshold']))
        refs.append(ip + ['threshold'])
        if it.get('is_default'):
            thr += ' (default)'
        mg = '---'
        if _is_num(it.get('margin')):
            mg = W.wrap(number_text(it['margin']))
            refs.append(ip + ['margin'])
        rows.append(_row([tex(it.get('name') or it.get('title') or ''), tex(it.get('metric') or ''), val,
                          thr, W.unit(it.get('unit') or ''), mg, tex(it.get('status_text') or '')], refs))
    head = ['Requirement', 'Metric', 'Value', 'Threshold', 'Unit', 'Margin', 'Status']
    verdict = tex(card.get('verdict_text') or '')
    return _table(f'Requirements scorecard of the rig ``{tex(card.get("variant") or "")}\'\': {verdict} '
                  '(a positive margin is good; ``(default)\'\' marks DexVantage\'s documented default threshold).',
                  'tab:scorecard', 'lllllrl', head, rows, wide=True)


def experiments_table(doc: dict, W: Writer) -> str:
    ep = ['path', 'results', 'experiments']
    frows = []
    for k, f in enumerate(doc.get('factors') or []):
        refs, cells = [], [tex(f.get('title') or f.get('path') or '')]
        for key in ('lo', 'hi', 'current'):
            v = f.get(key)
            if _is_num(v):
                cells.append(W.wrap(number_text(v)))
                refs.append(ep + ['factors', k, key])
            else:
                cells.append('---' if key != 'current' or v is None else tex(str(v))[:40])
        cells.append(W.unit(f.get('unit') or ''))
        frows.append(_row(cells, refs))
    rrows = []
    for mid, a in (doc.get('analysis') or {}).items():
        meta = next((r for r in doc.get('responses') or [] if r.get('id') == mid), {})
        refs, cells = [], [tex(meta.get('title') or mid)]
        ap = ep + ['analysis', mid]
        cells.append(W.wrap(number_text(a.get('n', 0))))
        refs.append(ap + ['n'])
        for key in ('mean', 'sd', 'min', 'max'):
            if _is_num(a.get(key)):
                cells.append(W.wrap(number_text(a[key])))
                refs.append(ap + [key])
            else:
                cells.append('---')
        cells.append(W.unit(meta.get('unit') or ''))
        rrows.append(_row(cells, refs))
    srows = []
    for mid, a in (doc.get('analysis') or {}).items():
        so = a.get('sobol') or {}
        for j, fpath in enumerate(so.get('factors') or []):
            sp = ep + ['analysis', mid, 'sobol']
            refs, cells = [], [tex(mid), tex(fpath)]
            for key in ('S1', 'ST'):
                vals, cis = so.get(key) or [], so.get(key + '_ci') or []
                v = vals[j] if j < len(vals) else None
                c = cis[j] if j < len(cis) else None
                if _is_num(v):
                    cell = W.wrap(number_text(v))
                    refs.append(sp + [key, j])
                    if isinstance(c, list) and len(c) == 2 and all(_is_num(x) for x in c):
                        cell += f' [{W.wrap(number_text(c[0]))}, {W.wrap(number_text(c[1]))}]'
                        refs += [sp + [key + '_ci', j, 0], sp + [key + '_ci', j, 1]]
                    cells.append(cell)
                else:
                    cells.append('---')
            srows.append(_row(cells, refs))
    name, how = tex(doc.get('name') or ''), tex(doc.get('method_label') or '')
    out = _table(f'Factors of the experiment ``{name}\'\' ({how}).',
                 'tab:experiment-factors', 'lrrrl', ['Factor', 'Low', 'High', 'Current', 'Unit'], frows)
    out += _table('Responses over the experiment\'s runs.', 'tab:experiment-responses', 'lrrrrrl',
                  ['Response', 'Runs', 'Mean', 'SD', 'Min', 'Max', 'Unit'], rrows)
    if srows:
        conf = next((a.get('sobol', {}).get('confidence') for a in (doc.get('analysis') or {}).values()
                     if a.get('sobol')), None)
        lv = f'{number_text(conf * 100)}\\,\\% ' if _is_num(conf) else ''
        out += _table(f'Sobol indices with their {lv}bootstrap intervals (first order $S_1$, total $S_T$).',
                      'tab:experiment-sobol', 'llll', ['Response', 'Factor', '$S_1$', '$S_T$'], srows)
    return out


def variants_table(project, results: dict, W: Writer) -> str:
    doc = results.get('variants_compare')
    if isinstance(doc, dict) and doc.get('variants') and doc.get('requirements'):
        vs = doc['variants']
        rows = []
        for j, rq in enumerate(doc['requirements']):
            refs, cells = [], [tex(rq.get('name') or rq.get('title') or '')]
            for i, _v in enumerate(vs):
                c = doc['cells'][i][j] if i < len(doc['cells']) and j < len(doc['cells'][i]) else {}
                if _is_num(c.get('value')):
                    cells.append(W.wrap(number_text(c['value'])) + f' ({tex(c.get("status") or "")})'.replace('_', ' '))
                    refs.append(['path', 'results', 'variants_compare', 'cells', i, j, 'value'])
                else:
                    cells.append('---')
            cells.append(W.unit(rq.get('unit') or ''))
            rows.append(_row(cells, refs))
        rows.append(r'\midrule')
        rows.append(_row(['Verdict'] + [tex(v.get('verdict_text') or '') for v in vs] + [''], []))
        head = ['Requirement'] + [tex(v.get('name') or '') for v in vs] + ['Unit']
        return _table('Rig variants against the requirements (value and status per variant).', 'tab:variants',
                      'l' + 'r' * len(vs) + 'l', head, rows, wide=len(vs) > 2)
    from . import library as LB
    rows = []
    for i, v in enumerate(project.variants):
        devs = sorted({(LB.get(s.model_id).title if s.model_id and LB.get(s.model_id) else 'default camera')
                       for s in v.sensors if s.enabled})
        rows.append(_row([tex(v.name), W.wrap(str(sum(1 for s in v.sensors if s.enabled))), tex(', '.join(devs))],
                         [['enabled', 'project', 'variants', i, 'sensors']]))
    return _table('Rig variants of the project (enabled sensors and their devices); run Variants compare for the '
                  'variants against the requirements.', 'tab:variants', 'lrl', ['Variant', 'Sensors', 'Devices'], rows)


def _bounded(W: Writer, v, bound) -> str:
    t = W.wrap(number_text(v))
    return (r'$\geq$ ' + t) if bound == 'at_least' else (r'$\leq$ ' + t) if bound == 'at_most' else t


def stats_table(doc: dict, W: Writer) -> str:
    """The Statistics study's comparisons: each side's figure, the difference with its interval, the test, the
    corrected p-value and the verdict word (core/studies/stats.py; rates in per cent, differences in points)."""
    rows = []
    for i, c in enumerate(doc.get('comparisons') or []):
        cp = ['path', 'results', 'stats', 'comparisons', i]
        refs, cells = [], [tex(c.get('title') or c.get('id') or ''), tex(c.get('test_label') or c.get('test') or '')]
        rates = c.get('kind') == 'rates'
        for side in ('a', 'b'):
            sd = c.get(side) or {}
            key = 'rate' if rates else 'mean'
            if _is_num(sd.get(key)):
                cells.append(W.wrap(number_text(sd[key] * (100.0 if rates else 1.0))))
                refs.append((['scaled', 100.0] + cp[1:] if rates else cp) + [side, key])
            else:
                cells.append('---')
        d = c.get('difference') or {}
        ci = d.get('ci') or []
        if _is_num(d.get('value')) and len(ci) == 2 and _is_num(ci[0]) and _is_num(ci[1]):
            sc = 100.0 if rates else 1.0
            pre = (['scaled', 100.0] + cp[1:]) if rates else cp
            cells.append(f'{W.wrap(number_text(d["value"] * sc))} [{W.wrap(number_text(ci[0] * sc))}, '
                         f'{W.wrap(number_text(ci[1] * sc))}]')
            refs += [pre + ['difference', 'value'], pre + ['difference', 'ci', 0], pre + ['difference', 'ci', 1]]
        else:
            cells.append('---')
        pa = c.get('p_adjusted')
        if _is_num(pa):
            cells.append(W.wrap(number_text(pa)))
            refs.append(cp + ['p_adjusted'])
        else:
            cells.append('---')
        cells.append('pp' if rates else W.unit(c.get('unit') or ''))
        cells.append(tex((c.get('verdict') or {}).get('word') or ''))
        rows.append(_row(cells, refs))
    corr = (doc.get('correction') or {}).get('label') or (doc.get('correction') or {}).get('method') or ''
    conf = next((c.get('confidence') for c in doc.get('comparisons') or [] if _is_num(c.get('confidence'))), None)
    head = ['Comparison', 'Test', 'A', 'B', 'B $-$ A [interval]', 'Adjusted p', 'Unit', 'Verdict']
    cap = ('Statistical comparisons (' + tex(doc.get('source_label') or doc.get('source') or '') + '): rates in '
           'per cent and their difference in percentage points; ' +
           (rf'{tex(number_text(round(100 * conf)))}\,\% intervals' if conf is not None else 'intervals') +
           (f'; p-values corrected across the comparisons ({tex(corr)})' if corr else '') + '.')
    return _table(cap, 'tab:statistics', 'llrrrrll', head, rows, wide=True,
                  note='A comparison without a clear difference does not show that the two are equal.')


def cells_table(doc: dict, W: Writer) -> str:
    """The multi-cell comparison: each figure of merit per cell and the geometric gap index to the reference cell
    (core/cells.py)."""
    cells = doc.get('cells') or []
    names = [str(c.get('name') or f'Cell {k + 1}') for k, c in enumerate(cells)]
    ref = doc.get('reference') or 0
    rows = []
    for r, m in enumerate((doc.get('metrics') or {}).get('rows') or []):
        mp = ['path', 'results', 'cells', 'metrics', 'rows', r]
        refs, out = [], [tex(m.get('title') or m.get('id') or '')]
        bounds = m.get('bounds') or []
        for k, v in enumerate(m.get('values') or []):
            if _is_num(v):
                out.append(_bounded(W, v, bounds[k] if k < len(bounds) else None))
                refs.append(mp + ['values', k])
            else:
                out.append('not stated')
        out.append(W.unit(m.get('unit') or ''))
        rows.append(_row(out, refs))
    pairs = (doc.get('gap') or {}).get('pairs') or []
    if pairs:
        by = {g.get('b'): j for j, g in enumerate(pairs)}
        refs, out = [], [f'Geometric gap index to {tex(names[ref] if ref < len(names) else "")}']
        for k, n in enumerate(names):
            j = by.get(n)
            if k == ref or j is None or not _is_num(pairs[j].get('total')):
                out.append('---')
            else:
                out.append(W.wrap(number_text(round(float(pairs[j]['total']), 3))))
                refs.append(['path', 'results', 'cells', 'gap', 'pairs', j, 'total'])
        out.append('')
        rows.append(r'\midrule')
        rows.append(_row(out, refs))
    head = ['Figure'] + [tex(n) + (' (reference)' if k == ref else '') for k, n in enumerate(names)] + ['Unit']
    return _table('Robot cells compared: the figures of merit of each cell and the geometric gap index of its cameras '
                  'to the reference cell (0 alike, 1 unrelated; a DexVantage heuristic, not a predicted success '
                  'rate).', 'tab:cells', 'l' + 'r' * len(names) + 'l', head, rows, wide=len(names) > 2)


def runs_table(doc: dict, W: Writer, limit: int = RUNS_ROWS) -> str:
    """The experiment tracker's runs (newest first, at most limit; core/studies/tracking.py): when, which study, the
    status and the headline figure."""
    from . import metrics as M
    M.load_metrics()
    heads = {m.study: m.id for m in M.METRICS.values() if 'headline' in m.tags}
    rows = []
    runs = doc.get('runs') or []
    for i, r in enumerate(runs[:limit]):
        h = r.get('headline') or {}
        mid = heads.get(r.get('study') or '')
        m = (r.get('metrics') or {}).get(mid) if mid else None
        refs, out = [], [r'\texttt{' + tex((r.get('run_id') or '')[:8]) + '}',
                         tex((r.get('started_utc') or '')[:16].replace('T', ' ')),
                         tex(r.get('title') or r.get('study') or ''), tex(r.get('status') or '')]
        if isinstance(m, dict) and _is_num(m.get('value')):
            out.append(tex(m.get('title') or mid))
            out.append(_bounded(W, m['value'], m.get('bound')))
            refs.append(['path', 'results', 'tracking', 'runs', i, 'metrics', mid, 'value'])
            out.append(W.unit(m.get('unit') or ''))
        else:
            out += ['---', '---', '']
        out.append(tex(h.get('word') or ''))
        rows.append(_row(out, refs))
    n = len(runs)
    more = f' (the newest {limit} of {n})' if n > limit else ''
    head = ['Run', 'Started (UTC)', 'Study', 'Status', 'Headline figure', 'Value', 'Unit', 'Word']
    return _table(f'Runs logged by the experiment tracker{more}, with each run\'s headline figure; '
                  'every run records its settings, seeds and provenance in the run store.', 'tab:runs', 'lllllrll',
                  head, rows, wide=True)


# ====================================================================== figures
def figure_specs(results: dict) -> list:
    """[(figure id, spec, study id)] for each result's per-item chart (report.figure) and an experiment's Sobol
    indices."""
    from . import figures as FG
    from .exporters import report as R
    out = []
    for sid, doc in results.items():
        f = R.figure(doc) if isinstance(doc, dict) else None
        if f and any(v is not None for v in f['values']):
            info = jobs.STUDIES.get(sid)
            unit = f['unit']
            spec = FG.bars([(lab, val, '#4477AA') for lab, val in zip(f['labels'], f['values'])], unit=unit,
                           title=f['title'][:1].upper() + f['title'][1:],
                           xlabel=f'{f["title"][:1].upper() + f["title"][1:]} ({unit})' if unit else f['title'],
                           source=f'{info.title if info else sid} result, {f["list"]}[].{f["key"]}')
            out.append((_label(f'{sid}-{f["key"]}'), spec, sid))
        if sid == 'experiments':
            for mid, a in (doc.get('analysis') or {}).items():
                so = a.get('sobol') or {}
                if so.get('ST') and so.get('factors'):
                    spec = FG.bars([(fp, v, '#4477AA') for fp, v in zip(so['factors'], so['ST'])],
                                   title=f'Total Sobol index of {mid}', xlabel='Total Sobol index',
                                   source=f'experiments result, analysis.{mid}.sobol.ST')
                    out.append((_label(f'experiments-sobol-{mid}'), spec, sid))
    return out


# ====================================================================== the kit
def _validation_doc(v):
    if v is None or isinstance(v, dict):
        return v
    try:
        with open(str(v), encoding='utf-8') as fh:
            d = json.load(fh)
    except (OSError, ValueError) as e:
        raise PublicationError(f'The validation report {os.path.basename(str(v))} cannot be read ({e}).') from None
    if not isinstance(d, dict) or d.get('schema') != 'dexvantage.validation/1':
        raise PublicationError(f'{os.path.basename(str(v))} is not a DexVantage validation report (validation.json).')
    return d


def _scorecard(project, results: dict, cache, allow_run: bool):
    """(card or None, studies it still needs)."""
    doc = results.get('requirements')
    if isinstance(doc, dict) and isinstance(doc.get('items'), list):
        return doc, []
    if not project.requirements:
        return None, []
    need = sorted({r.study for r in project.requirements} - set(results))
    if need and not allow_run:
        return None, need
    from . import requirements as RQ
    try:
        return RQ.check(project, cache=cache), []
    except StudyError:
        return None, need


def run_plan(project, results: dict, figs: list, card) -> dict:
    """The run plan (core/run_plan.py) that regenerates every result, figure and table of the kit."""
    steps = []
    for sid in results:
        if sid in ('summary', 'requirements'):
            continue
        st = {'study': sid, 'out': f'results/{sid}.json'}
        if project.analysis.studies.get(sid):
            st['settings'] = json.loads(json.dumps(project.analysis.studies[sid]))
        steps.append(st)
    if card is not None and project.requirements:
        steps.append({'check': [r.to_dict() for r in project.requirements], 'out': 'results/scorecard.json'})
    steps.append({'export': '.', 'formats': ['paper_kit'],
                  'options': {'paper_kit': {'results_dir': 'results', 'studies': []}}})
    return {'name': f'paper-kit-{_label(project.name)}'[:32], 'steps': steps}


def appendix_tex(project, results: dict, refused: dict, figs: list, plan: dict, env: dict, W: Writer) -> str:
    from . import provenance as PV
    dv = env.get('dexvantage') or {}
    rt = env.get('runtime') or {}
    git = dv.get('git') or {}
    fname = os.path.basename(project.path) if project.path else 'PROJECT.dexv'
    L = [r'\section*{Reproducibility}',
         'Every result, table and figure of this kit is regenerated by the run plan '
         '\\texttt{reproduce/run\\_plan.json} (DexVantage\'s \\texttt{--run-plan}), which runs each study with '
         'the settings listed and then writes the '
         'kit again from those results. The studies are deterministic given their seeds; bit-for-bit agreement '
         'needs the same code and library versions (Table~\\ref{tab:versions}), and \\texttt{--reproduce} compares a '
         're-run with a recorded result number by number.',
         '\\begin{verbatim}\n' + f'dexvantage {fname} --run-plan reproduce/run_plan.json out/\n' + '\\end{verbatim}']
    srows = []
    for i, st in enumerate(plan['steps']):
        kind = next(k for k in ('study', 'check', 'export') if k in st)
        what = st[kind] if isinstance(st[kind], str) else f'{len(st[kind])} requirements'
        made = [fid for fid, _s, sid in figs if kind == 'study' and sid == st.get('study')]
        srows.append(_row([str(i + 1), tex(kind), f'\\texttt{{{tex(what)}}}', tex(st.get('out') or ''),
                           tex(', '.join(made) or '---')], []))
    L.append(_table('The run plan\'s steps and the figures each one regenerates.', 'tab:plan', 'rllll',
                    ['Step', 'Kind', 'What', 'Output', 'Figures'], srows))
    seeds = []
    for sid in results:
        for k, v in PV.seeds_of(PV.study_settings(project, sid)).items():
            seeds.append(_row([tex(sid), f'\\texttt{{{tex(k)}}}', tex(str(v))], []))
        if sid == 'tolerance' and _is_num(results[sid].get('seed')):
            seeds.append(_row([tex(sid), '\\texttt{seed}', tex(str(results[sid]['seed']))], []))
    if seeds:
        L.append(_table('Random seeds of the studies (each generator is NumPy\'s PCG64 built from the seed).',
                        'tab:seeds', 'lll', ['Study', 'Setting', 'Seed'], seeds))
    else:
        L.append('None of the studies reported here draws random numbers (no seed setting).')
    vrows = [_row(['DexVantage', tex(dv.get('version') or '---')], []),
             _row(['Code SHA-256', f'\\texttt{{{tex((dv.get("code_sha256") or "not available")[:16])}}}'], [])]
    if git.get('commit'):
        vrows.append(_row(['Commit', f'\\texttt{{{tex(git["commit"][:12])}}}'
                           + (' (with local changes)' if git.get('dirty') else '')], []))
    for k in ('python', 'numpy', 'scipy', 'opencv', 'mujoco', 'qt', 'pyqt', 'blas', 'platform', 'machine'):
        if rt.get(k):
            vrows.append(_row([tex(k), tex(str(rt[k]))], []))
    L.append(_table('Software versions of the run that wrote this kit.', 'tab:versions', 'll',
                    ['Component', 'Version'], vrows))
    hrows = []
    for sid, doc in results.items():
        pv = doc.get('provenance') if isinstance(doc.get('provenance'), dict) else None
        sha = (pv or {}).get('result_sha256') or PV.result_digest({k: v for k, v in doc.items()
                                                                    if k not in ('cached', 'revision')})
        hrows.append(_row([tex(sid), f'\\texttt{{{sha[:16]}}}', tex((pv or {}).get('created_utc') or 'this export')],
                          []))
    L.append(_table('Result hashes (SHA-256 of each result\'s canonical JSON without its provenance, first 16 '
                    'hexadecimal digits).', 'tab:hashes', 'lll', ['Study', 'Result SHA-256', 'Created (UTC)'],
                    hrows))
    if refused:
        L.append('Studies that could not run: ' + '; '.join(f'{tex(jobs.STUDIES[k].title if k in jobs.STUDIES else k)}'
                                                            f' ({tex(r)})' for k, r in refused.items()) + '.')
    return '\n\n'.join(L) + '\n'


def main_tex(st: dict, project, parts: dict, figs: list, fig_files: dict) -> str:
    title = st.get('title') or f'{project.name}: sensing rig'
    L = [r'% Written by DexVantage (publication kit). Compile: pdflatex main, bibtex main, pdflatex main twice.',
         r'\documentclass[10pt]{article}', r'\usepackage[T1]{fontenc}', r'\usepackage[utf8]{inputenc}',
         r'\usepackage{booktabs}', r'\usepackage{graphicx}', r'\usepackage{url}']
    if st['siunitx']:
        L.append(r'\usepackage{siunitx}')
    L.append(DVNUM)
    if st['bibliography'] == 'biblatex':
        L += [r'\usepackage[backend=biber]{biblatex}', r'\addbibresource{refs.bib}']
    L += [f'\\title{{{tex(title)}}}', r'\author{}', r'\date{}', r'\begin{document}', r'\maketitle']
    if 'methods' in parts:
        L += [r'\section{Methods}', r'\input{methods}']
    for name in TABLE_PARTS:
        if name in parts:
            L.append(f'\\input{{tables/{name}}}')
    if 'figures' in st['sections']:
        for fid, spec, _sid in figs:
            f = fig_files.get(fid, {})
            L += [r'\begin{figure}[t]', r'\centering']
            if f.get('pdf'):
                L.append(f'\\includegraphics[width=\\columnwidth]{{{f["pdf"]}}}')
            else:
                L.append(r'\fbox{\parbox{0.9\columnwidth}{The PDF of this figure was not written; its SVG is '
                         f'\\texttt{{{tex(f.get("svg") or fid)}}}.}}}}')
            cap = tex(spec.get('title') or fid)
            L += [f'\\caption{{{cap} ({tex(spec.get("source") or "")}).}}', f'\\label{{fig:{fid}}}', r'\end{figure}']
    if st['bibliography'] == 'biblatex':
        L.append(r'\printbibliography')
    else:
        L += [r'\bibliographystyle{plain}', r'\bibliography{refs}']
    if 'appendix' in parts:
        L += [r'\appendix', r'\input{appendix}']
    L.append(r'\end{document}')
    return '\n'.join(L) + '\n'


def build(project, options: dict | None = None, ctx=None) -> dict:
    """The kit as data: its TeX texts, figure specs, BibTeX, run plan, trace and checks (module docstring)."""
    from . import provenance as PV
    from .exporters import report as R
    o = dict(options or {})
    st = settings_of(project, {k: v for k, v in o.items() if k in SETTING_KEYS})
    cache = jobs.ResultCache(64)
    ropts = {k: o[k] for k in ('results', 'results_dir', 'studies') if k in o}
    results, refused = R.collect(project, ropts, ctx, cache)
    results = {k: {kk: vv for kk, vv in v.items() if kk not in ('cached', 'revision')} for k, v in results.items()}
    if 'runs' in st['sections'] and 'tracking' not in results:
        tr = _tracker_summary(project)
        if tr is not None:
            results['tracking'] = tr
    for sid, doc in results.items():
        if sid in jobs.STUDIES and not doc.get('cancelled'):
            try:
                cache.put(cache.key(sid, project), {k: v for k, v in doc.items() if k != 'provenance'})
            except Exception:                                 # noqa: BLE001 — the scorecard runs the study instead
                pass
    validation = _validation_doc(o.get('validation'))
    allow = bool(ropts.get('studies')) or ('studies' not in ropts and 'results' not in ropts
                                           and 'results_dir' not in ropts)
    card, missing = _scorecard(project, results, cache, allow)
    source = source_data(project, results, validation, card)
    cm = CiteMap(project)
    W = Writer(st['siunitx'])
    parts: dict = {}
    if 'methods' in st['sections']:
        parts['methods'] = '% Written by DexVantage (publication kit): every number is traced in provenance.json.\n' \
            + DVNUM + '\n' + methods(project, results, source, cm, W, st, card, refused)
    trace = list(W.trace)
    TW = Writer(st['siunitx'])
    TW.cited = W.cited
    if 'sensors' in st['sections']:
        parts['sensors'] = sensors_table(project, TW)
    if 'results' in st['sections'] and any(s not in META_TABLES or s == 'tolerance' for s in results):
        parts['results'] = results_table(results, TW)
    if 'scorecard' in st['sections'] and (card is not None or project.requirements):
        parts['scorecard'] = scorecard_table(card, TW, missing)
    if 'experiments' in st['sections'] and isinstance(results.get('experiments'), dict) and \
            results['experiments'].get('factors'):
        parts['experiments'] = experiments_table(results['experiments'], TW)
    if 'variants' in st['sections'] and (len(project.variants) > 1 or 'variants_compare' in results):
        parts['variants'] = variants_table(project, results, TW)
    if 'statistics' in st['sections'] and isinstance(results.get('stats'), dict) and \
            results['stats'].get('comparisons'):
        parts['statistics'] = stats_table(results['stats'], TW)
    if 'cells' in st['sections'] and isinstance(results.get('cells'), dict) and \
            (results['cells'].get('metrics') or {}).get('rows'):
        parts['cells'] = cells_table(results['cells'], TW)
    if 'runs' in st['sections'] and isinstance(results.get('tracking'), dict) and results['tracking'].get('runs'):
        parts['runs'] = runs_table(results['tracking'], TW)
    figs = figure_specs(results) if 'figures' in st['sections'] else []
    for k, spec in enumerate(o.get('figure_specs') or []):
        if isinstance(spec, dict) and spec.get('kind') in ('curves', 'bars') and 'figures' in st['sections']:
            figs.append((_label(f'page-{k + 1}-{spec.get("title") or "chart"}')[:48], spec, ''))
    plan = run_plan(project, results, figs, card)
    if 'appendix' in st['sections']:
        parts['appendix'] = appendix_tex(project, results, refused, figs, plan, source['environment'], TW)
    keys = list(W.cited)
    bib = cm.bib(project, keys)
    checks = {'numbers': verify_numbers(parts.get('methods', ''), trace, source),
              'keys': {'used': sorted(keys), 'defined': sorted(bib_keys(bib))},
              'latex': {k: latex_problems(v) for k, v in parts.items()},
              'tables': verify_tables({k: v for k, v in parts.items() if k != 'methods'}, source)}
    used = set(cited_keys('\n'.join(parts.values())))
    checks['keys']['missing'] = sorted(used - set(checks['keys']['defined']))
    checks['keys']['unused'] = sorted(set(checks['keys']['defined']) - used)
    v = project.variant
    return {'schema': SCHEMA, 'title': st.get('title') or f'{project.name}: sensing rig', 'project': project.name,
            'variant': v.name, 'settings': stored_settings(st), 'siunitx': st['siunitx'],
            'sections': st['sections'], 'parts': parts, 'bib': bib, 'figures': figs, 'plan': plan,
            'results': {sid: {'study': sid, 'seconds': d.get('seconds'), 'cancelled': bool(d.get('cancelled')),
                              'result_sha256': ((d.get('provenance') or {}).get('result_sha256')
                                                if isinstance(d.get('provenance'), dict) else None)
                              or PV.result_digest(d)} for sid, d in results.items()},
            'refused': refused, 'scorecard_missing': missing, 'trace': trace, 'checks': checks,
            'status': status(checks), '_st': st, '_source': source}


def _tracker_summary(project) -> dict | None:
    """The experiment tracker's summary (the 'tracking' study: reading the run store is quick) when the project's
    tracking is on and its store holds runs; None otherwise."""
    from . import tracking as TR
    try:
        if not TR.enabled(project) or not TR.store_for(project).exists():
            return None
        doc = jobs.run_study('tracking', project.snapshot())
    except StudyError:
        return None
    doc = {k: v for k, v in doc.items() if k not in ('cached', 'revision')}
    return doc if doc.get('runs') else None


_DVSRC = re.compile(r'%\s*dv-src:\s*(\[.*\])\s*$')
_CELLNUM = re.compile(r'\\(?:num|dvnum)\{((?:[^{}]|\{[^{}]*\})*)\}')


def cell_number(s: str) -> float:
    s = s.replace(r'\ensuremath', '').strip('{} ')
    m = re.fullmatch(r'(-?[\d.]+)\\times10\^\{(-?\d+)\}', s)
    if m:
        return float(f'{m.group(1)}e{m.group(2)}')
    return float(s)


def resolve_ref(source: dict, ref: list):
    """The source value a table cell names: ['metric', study, metric id], ['path', ...], ['scaled', k, ...] or
    ['enabled', ...] (the count of enabled sensors at that path)."""
    from . import metrics as M
    if ref[0] == 'metric':
        M.load_metrics()
        return M.read(M.get(ref[2]), source['results'][ref[1]])['value']
    if ref[0] == 'scaled':
        return float(ref[1]) * float(resolve(source, ref[2:]))
    if ref[0] == 'enabled':
        return sum(1 for s in resolve(source, ref[1:]) if s.get('enabled', True))
    return resolve(source, ref[1:])


def verify_tables(parts: dict, source: dict) -> dict:
    """Every number cell of every table row matched to its source value (the row's dv-src comment)."""
    n, problems = 0, []
    for name, text in parts.items():
        for ln, line in enumerate(text.split('\n'), 1):
            m = _DVSRC.search(line)
            body = re.sub(r'(?<!\\)%.*$', '', line)
            cells = _CELLNUM.findall(body)
            if not m:
                if cells and r'\\' in body:
                    problems.append(f'{name} line {ln}: numbers without a source')
                continue
            refs = json.loads(m.group(1))
            if len(refs) != len(cells):
                problems.append(f'{name} line {ln}: {len(cells)} numbers, {len(refs)} sources')
                continue
            for c, ref in zip(cells, refs):
                n += 1
                try:
                    v = resolve_ref(source, ref)
                    t = cell_number(c)
                except (KeyError, IndexError, TypeError, ValueError, StudyError):
                    problems.append(f'{name} line {ln}: {c} has no source value')
                    continue
                txt = number_text(t)
                d = len(c.split('.', 1)[1].split('\\')[0].split('e')[0]) if '.' in c else 0
                if not (_is_num(v) and abs(float(v) - t) <= max(0.5 * 10 ** (-d), 5e-4 * abs(t)) * (1 + 1e-9)
                        + 1e-12):
                    problems.append(f'{name} line {ln}: {txt} against the source {v}')
    return {'numbers': n, 'problems': problems}


def status(checks: dict) -> dict:
    """The page's headline: the numbers traced, and whether everything checks."""
    nb = checks['numbers']
    bad = (len(nb['problems']) + len(checks['keys']['missing']) + len(checks['keys']['unused'])
           + sum(len(v) for v in checks['latex'].values()) + len(checks['tables']['problems']))
    if bad:
        first = (nb['problems'] + [f'citation {k} has no BibTeX entry' for k in checks['keys']['missing']]
                 + [f'{k}: {p}' for k, v in checks['latex'].items() for p in v] + checks['tables']['problems']
                 + [f'BibTeX entry {k} is not cited' for k in checks['keys']['unused']])[0]
        return {'level': 'critical', 'word': 'check', 'traced': nb['traced'], 'numbers': nb['numbers'],
                'text': f'{bad} problem{"s" if bad != 1 else ""} in the kit: {first}.'}
    return {'level': 'good', 'word': 'ready', 'traced': nb['traced'], 'numbers': nb['numbers'],
            'text': (f'All {nb["numbers"]} numbers in the methods paragraph and {checks["tables"]["numbers"]} in the '
                     f'tables trace to the project or a result; every citation has its BibTeX entry and the LaTeX is '
                     'balanced.')}


def public(kit: dict) -> dict:
    """The kit without its working state (what provenance.json holds)."""
    out = {k: v for k, v in kit.items() if not k.startswith('_') and k not in ('parts', 'bib', 'figures')}
    out['figures'] = [{'id': fid, 'title': spec.get('title'), 'source': spec.get('source'), 'study': sid}
                      for fid, spec, sid in kit['figures']]
    out['files'] = kit.get('files', [])
    return out


def write(project, out_dir: str, options: dict | None = None, ctx=None) -> dict:
    """Write the kit under out_dir/paper_kit/: main.tex, methods.tex, tables/*.tex, figures/*, refs.bib,
    appendix.tex, reproduce/run_plan.json and provenance.json. {'files' (relative to out_dir), 'notes', 'kit'}."""
    from . import figures as FG
    from .exporters import _common as C
    o = dict(options or {})
    kit = build(project, o, ctx)
    st = kit['_st']
    if not kit['parts'] and not kit['figures']:
        raise PublicationError('There is nothing to put in the kit: choose at least one section.')
    files, notes = [], []
    for name, text in kit['parts'].items():
        rel = f'{FOLDER}/{name}.tex' if name in ('methods', 'appendix') else f'{FOLDER}/tables/{name}.tex'
        files.append(C.write_text(out_dir, rel, text))
    fig_files: dict = {}
    pdf_fail = ''
    for k, (fid, spec, _sid) in enumerate(kit['figures']):
        if ctx is not None:
            ctx.report(0.8 + 0.15 * k / max(1, len(kit['figures'])), f'Figure {fid}')
        for w in st['widths']:
            for fmt in ('svg', 'pdf') if st['pdf_figures'] else ('svg',):
                rel = f'{FOLDER}/figures/{fid}-{w}.{fmt}'
                path = os.path.join(out_dir, rel)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                h = None
                sp = spec
                if spec.get('kind') == 'bars':
                    h = max(FG.size_mm(w)[1], 10.0 + 5.0 * len(spec.get('rows') or []))
                    sp = fit_labels(spec, w)
                try:
                    FG.write(sp, path, fmt=fmt, width=w, height_mm=h, meta={'paper_kit': fid})
                except (StudyError, ImportError, RuntimeError) as e:
                    if fmt == 'pdf':
                        pdf_fail = str(e)
                        continue
                    notes.append(f'Figure {fid} ({w} column) could not be drawn: {e}')
                    continue
                files.append(rel)
                fig_files.setdefault(fid, {})
                if w == st['widths'][0]:
                    fig_files[fid][fmt] = rel[len(FOLDER) + 1:]
    if pdf_fail:
        notes.append(f'PDF figures were not written ({pdf_fail}); the SVGs are, and main.tex says where.')
    main = main_tex(st, project, kit['parts'], kit['figures'], fig_files)
    kit['parts_main'] = main
    files.append(C.write_text(out_dir, f'{FOLDER}/main.tex', main))
    files.append(C.write_text(out_dir, f'{FOLDER}/refs.bib', kit['bib']))
    files.append(C.write_json(out_dir, f'{FOLDER}/reproduce/run_plan.json', kit['plan']))
    kit['checks']['latex']['main'] = latex_problems(main)
    kit['checks']['inputs'] = missing_inputs(os.path.join(out_dir, FOLDER), main)
    kit['status'] = status(kit['checks'])
    if kit['checks']['inputs']:
        kit['status'] = {**kit['status'], 'level': 'critical', 'word': 'check',
                         'text': 'main.tex names files the kit did not write: ' + ', '.join(kit['checks']['inputs'])}
    kit['files'] = [f[len(FOLDER) + 1:] for f in files]
    files.append(C.write_json(out_dir, f'{FOLDER}/provenance.json', public(kit)))
    s = kit['status']
    notes.insert(0, f'{len(kit["parts"])} TeX part(s), {len(kit["figures"])} figure(s) at '
                    f'{" and ".join(st["widths"])} column width, {len(bib_keys(kit["bib"]))} BibTeX entries; '
                    f'{s["word"]}: {s["text"]}')
    if kit['refused']:
        notes.append(f'{len(kit["refused"])} stud{"y" if len(kit["refused"]) == 1 else "ies"} could not run (listed '
                     'in the appendix).')
    if kit['scorecard_missing']:
        notes.append('The scorecard was not evaluated: it needs ' + ', '.join(kit['scorecard_missing']) + '.')
    return {'files': files, 'notes': notes, 'kit': kit}


def fit_labels(spec: dict, width) -> dict:
    """A bars spec whose row labels are cut (with an ellipsis) to at most 40 % of the figure's width at the figure
    font's estimated character width (core/figures text_w), so a long sensor name leaves room for the bars."""
    from . import figures as FG
    wmm, _h = FG.size_mm(width)
    n = max(8, int(0.4 * wmm * FG.PT_PER_MM / (FG.FONT_PT * FG.CHAR_EM)))
    rows = [dict(r, label=r['label'] if len(r['label']) <= n else r['label'][:n - 1].rstrip() + '…')
            for r in spec.get('rows') or []]
    return dict(spec, rows=rows)


def missing_inputs(folder: str, main: str) -> list:
    """Files main.tex \\input's or \\includegraphics that are not in the kit's folder."""
    out = []
    for m in re.finditer(r'\\(input|includegraphics)(?:\[[^\]]*\])?\{([^{}]+)\}', strip_comments(main)):
        rel = m.group(2)
        cand = [rel] if m.group(1) == 'includegraphics' else [rel, rel + '.tex']
        if not any(os.path.isfile(os.path.join(folder, c)) for c in cand):
            out.append(rel)
    return out


def latex_engine() -> str | None:
    """The first LaTeX engine on PATH (pdflatex, lualatex, tectonic), or None."""
    import shutil
    for e in ('pdflatex', 'lualatex', 'tectonic'):
        if shutil.which(e):
            return e
    return None
