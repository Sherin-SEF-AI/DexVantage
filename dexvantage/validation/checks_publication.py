"""Checks of the publication kit (core/publication.py, the 'paper_kit' exporter), each against a reference built
independently here. Every check writes the kit of the SO-100 sample (the project the suite was given, else the
sample itself) from the results of five studies run here (rig inventory, bandwidth, coverage, time sync, motion
blur) plus a grasp result written here so the interval column is exercised (a grasp study run takes tens of
seconds; the kit reports results as given).

- publication.methods_numbers: every number in methods.tex, found with a pattern written here, looked for in the
  source data read here: the project's own JSON (Project.to_dict), the library entry files of its devices read from
  disk, the results, the software versions (importlib.metadata and the package's __version__), the number of checks
  each validation model registers, and the length of every list in those. A number counts as found when a source
  number rounds to it at the digits shown, or when it is a word of a source string (a name such as "Camera 2", a
  version). The error counts the numbers not found; at least 20 numbers must be in the paragraph. Tolerance 0: a
  number has a source or not. It is a membership test over a large pool (every list length and every number of the
  project, the device files and the results): it catches a number with no source anywhere, not one read from the
  wrong place (a small integer is almost always in the pool); the paragraph's per-number trace (verify_numbers) and
  the tests' planted numbers cover that.
- publication.bibtex_keys: the citation keys of every .tex file of the kit (\\cite{...} read here) against the keys
  refs.bib defines, read with the brace-counting BibTeX parser of checks_versioning. The error counts the keys in one
  set and not the other plus entries that do not parse. Tolerance 0.
- publication.latex_structure: every .tex file scanned here character by character: braces balance (an escaped
  brace and comments do not count), every \\begin has its \\end in the right order (verbatim content skipped), every
  \\input and \\includegraphics names a file the kit wrote, and no table has a vertical rule (a '|' in its column
  specification) or \\hline (booktabs' first guideline, Fear 2020). The error counts the problems. Tolerance 0.
- publication.table_numbers: every number cell of every table (\\num{...} with siunitx, \\dvnum{...} without; both
  settings are written) against the value its row names: a study's named figure read with core.metrics.read on the
  result, or the value at a path of the project, a library file or a result read here (scaled where the row says).
  The error counts cells that differ by more than half a unit of their last digit (or 5 × 10⁻⁴ of the value, the
  four significant digits the tables write) plus rows with numbers and no source. Tolerance 0.
"""
from __future__ import annotations

import json
import os
import re

from .suite import Outcome, check

_FIXED = {'grasp_visibility': {'schema': 'dexvantage.grasp_visibility/1', 'study': 'grasp_visibility',
                               'summary': {'cvs_pct': 96.5, 'ci_low': 0.931, 'ci_high': 0.987, 'ci_level': 0.95,
                                           'ci_method': 'percentile bootstrap (written by the check)',
                                           'grasps': 40, 'feasible': 38, 'both_fingertips_rate': 0.9}}}


def _project(ctx):
    from ..core.robot.samples import sample_project
    return sample_project('so100') if not getattr(ctx, 'given', False) else ctx.project


def _kit(ctx, siunitx=False):
    """(folder of the kit, results, project), cached per setting on the check context."""
    key = ('publication', siunitx)
    if key in ctx.cache:
        return ctx.cache[key]
    from ..core import exporters as X
    from ..core import jobs
    p = _project(ctx)
    res = {}
    for sid in ('rig', 'bandwidth', 'coverage', 'time_sync', 'motion_blur'):
        try:
            res[sid] = jobs.run_study(sid, p.snapshot())
        except jobs.StudyError:
            pass
    res.update(json.loads(json.dumps(_FIXED)))
    out = os.path.join(ctx.tmp(), f'kit-{int(siunitx)}')
    man = X.export(p, out, ['paper_kit'], {'paper_kit': {'results': res, 'studies': [], 'siunitx': siunitx,
                                                         'pdf_figures': False}})
    if man['formats']['paper_kit']['status'] != 'ok':
        raise RuntimeError(man['formats']['paper_kit'].get('reason'))
    ctx.cache[key] = (os.path.join(out, 'paper_kit'), res, p)
    return ctx.cache[key]


def _read(path):
    with open(path, encoding='utf-8') as fh:
        return fh.read()


def _tex_files(folder):
    out = []
    for root, _d, files in os.walk(folder):
        for f in files:
            if f.endswith('.tex'):
                out.append(os.path.join(root, f))
    return sorted(out)


def _uncomment(text):
    return '\n'.join(re.sub(r'(?<!\\)%.*$', '', ln) for ln in text.split('\n'))


# ====================================================================== the numbers of the paragraph
_TOKEN = re.compile(r'(?<![A-Za-z0-9_.\\])-?[0-9]+(?:\.[0-9]+)*(?![A-Za-z0-9_]|\.[0-9])')


def paragraph_numbers(tex: str) -> list:
    """The numbers of a TeX paragraph: comments, macro definitions, citation keys, labels and references dropped."""
    t = _uncomment(tex)
    t = re.sub(r'^\\providecommand.*$', '', t, flags=re.M)
    t = re.sub(r'\\(?:cite|ref|label)\{[^{}]*\}', ' ', t)
    return _TOKEN.findall(t)


def _walk(o, nums, strs, lens):
    if isinstance(o, dict):
        lens.add(len(o))
        for v in o.values():
            _walk(v, nums, strs, lens)
    elif isinstance(o, (list, tuple)):
        lens.add(len(o))
        for v in o:
            _walk(v, nums, strs, lens)
    elif isinstance(o, bool):
        return
    elif isinstance(o, (int, float)):
        nums.append(float(o))
    elif isinstance(o, str):
        strs.add(o)


def source_pool(project, results: dict):
    """(numbers, string words, list lengths) of everything the paragraph may draw on (module docstring)."""
    import importlib.metadata as md

    from .. import __version__
    from ..core import library as LB
    from . import suite as VS
    nums, strs, lens = [], set(), set()
    pd = project.to_dict()
    pd.pop('log', None)
    _walk(pd, nums, strs, lens)
    _walk(results, nums, strs, lens)
    for s in {s.model_id for v in project.variants for s in v.sensors if s.model_id}:
        e = LB.get(s)
        if e is not None and e.path:
            _walk(json.loads(_read(e.path)), nums, strs, lens)
    _walk({'resolution': list(LB.DEFAULT_RESOLUTION), 'h': LB.DEFAULT_HFOV_DEG, 'v': LB.DEFAULT_VFOV_DEG}, nums,
          strs, lens)
    strs.add(__version__)
    for pkg in ('mujoco', 'numpy'):
        try:
            strs.add(md.version(pkg))
        except md.PackageNotFoundError:
            pass
    import platform
    strs.add(platform.python_version())
    VS.load_checks()
    per = {}
    for sp in VS.CHECKS:
        per[sp.model] = per.get(sp.model, 0) + 1
    nums += [float(v) for v in per.values()]
    words = set()
    for s in strs:
        words.update(_TOKEN.findall(s))
    return nums, words, lens


def found(token: str, nums, words, lens) -> bool:
    if token in words:
        return True
    if token.count('.') > 1:          # a version-like token not in the word set (float() would raise)
        return False
    t = float(token)
    d = len(token.split('.')[1]) if token.count('.') == 1 else 0
    half = 0.5 * 10 ** (-d) * (1 + 1e-9) + 1e-12
    if d == 0 and float(int(t)) in {float(x) for x in lens}:
        return True
    return any(abs(v - t) <= half for v in nums)


@check('publication.methods_numbers', 'publication', 'Methods paragraph: every number from the source data',
       'numbers in the paragraph not found in the project, the library files, the results, the versions or the '
       'check registry (rounded to the digits shown, or a word of a source string, or a list length)',
       'the source data read here (Project.to_dict, the library JSON files, the result documents, '
       'importlib.metadata versions, the check registry)', 0, 'numbers', refs=('latex2e_unofficial',),
       cost='study')
def _numbers(ctx):
    folder, res, p = _kit(ctx)
    toks = paragraph_numbers(_read(os.path.join(folder, 'methods.tex')))
    nums, words, lens = source_pool(p, res)
    missing = [t for t in toks if not found(t, nums, words, lens)]
    return Outcome(len(missing), {'numbers': len(toks), 'not_found': missing[:10]}, ok=len(toks) >= 20,
                   note='A paragraph with fewer than 20 numbers would not exercise the check.')


# ====================================================================== citations
@check('publication.bibtex_keys', 'publication', 'Paper kit: cited keys equal the BibTeX keys',
       'keys cited in the .tex files and not defined in refs.bib, plus keys defined and not cited, plus entries '
       'that do not parse', 'the \\cite{} keys read here; refs.bib read with the brace-counting parser of '
       'checks_versioning', 0, 'keys', refs=('bibtex_btxdoc',), cost='study')
def _keys(ctx):
    from .checks_versioning import parse_bibtex
    folder, _res, _p = _kit(ctx)
    cited = set()
    for f in _tex_files(folder):
        for m in re.finditer(r'\\cite\{([^{}]*)\}', _uncomment(_read(f))):
            cited |= {k.strip() for k in m.group(1).split(',') if k.strip()}
    bib = _read(os.path.join(folder, 'refs.bib'))
    try:
        ents = parse_bibtex(bib)
    except (ValueError, IndexError) as e:
        return Outcome(1.0 + len(cited), {'parse_error': str(e)})
    defined = {k for _t, k, _f in ents}
    heads = len(re.findall(r'^@', bib, re.M))
    return Outcome(len(cited - defined) + len(defined - cited) + abs(heads - len(ents)),
                   {'cited': len(cited), 'defined': len(defined), 'undefined': sorted(cited - defined)[:5],
                    'uncited': sorted(defined - cited)[:5]}, ok=len(cited) >= 3)


# ====================================================================== LaTeX structure
def structure_problems(text: str, folder: str) -> list:
    """Structural problems of a TeX file (module docstring), found by a scan written here."""
    probs, stack, depth = [], [], 0
    lines = text.split('\n')
    verb = None
    for n, raw in enumerate(lines, 1):
        if verb is not None:
            if ('\\end{' + verb + '}') in raw:
                verb = None
            continue
        k = 0
        while k < len(raw):
            c = raw[k]
            if c == '%':
                break
            if c == '\\':
                rest = raw[k:]
                m = re.match(r'\\(begin|end)\{([^{}]+)\}', rest)
                if m:
                    if m.group(1) == 'begin':
                        stack.append(m.group(2))
                        if m.group(2) == 'verbatim':
                            stack.pop()
                            verb = 'verbatim'
                            break
                        if m.group(2) == 'tabular':
                            spec = re.match(r'\\begin\{tabular\}\{([^{}]*)\}', rest)
                            if spec and '|' in spec.group(1):
                                probs.append(f'line {n}: a vertical rule in a table')
                    else:
                        if not stack or stack[-1] != m.group(2):
                            probs.append(f'line {n}: \\end{{{m.group(2)}}} does not close '
                                         f'{stack[-1] if stack else "anything"}')
                        if stack:
                            stack.pop()
                    k += len(m.group(0))
                    continue
                m = re.match(r'\\(input|includegraphics)(?:\[[^\]]*\])?\{([^{}]+)\}', rest)
                if m:
                    p = os.path.join(folder, m.group(2))
                    if not (os.path.isfile(p) or os.path.isfile(p + '.tex')):
                        probs.append(f'line {n}: {m.group(2)} is not in the kit')
                if rest.startswith('\\hline'):
                    probs.append(f'line {n}: \\hline (booktabs rules only)')
                k += 2
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth < 0:
                    probs.append(f'line {n}: unbalanced closing brace')
                    depth = 0
            k += 1
    if depth:
        probs.append(f'{depth} unclosed brace(s)')
    probs += [f'{s} never ended' for s in stack]
    if verb is not None:
        probs.append('verbatim never ended')
    return probs


@check('publication.latex_structure', 'publication', 'Paper kit: LaTeX structure',
       'unbalanced braces, \\begin/\\end mismatches, \\input or \\includegraphics of files not written, vertical '
       'rules or \\hline in tables, over every .tex file of the kit (both unit settings)',
       'a character-by-character scan written here', 0, 'problems',
       refs=('latex2e_unofficial', 'booktabs_fear2020'), cost='study')
def _structure(ctx):
    probs, files = [], 0
    for si in (False, True):
        folder, _res, _p = _kit(ctx, si)
        for f in _tex_files(folder):
            files += 1
            probs += [f'{os.path.relpath(f, folder)} {p}' for p in structure_problems(_read(f), folder)]
    return Outcome(len(probs), {'files': files, 'problems': probs[:10]}, ok=files >= 8)


# ====================================================================== table numbers
_CELL = re.compile(r'\\(?:num|dvnum)\{((?:[^{}]|\{[^{}]*\})*)\}')


def _value(c: str) -> tuple:
    c = c.strip()
    m = re.fullmatch(r'(-?[0-9.]+)\\times10\^\{(-?[0-9]+)\}', c)
    if m:
        return float(m.group(1)) * 10 ** int(m.group(2)), len(m.group(1).split('.')[1]) if '.' in m.group(1) else 0
    m = re.fullmatch(r'(-?[0-9.]+)(?:e(-?[0-9]+))?', c)
    v = float(m.group(1)) * (10 ** int(m.group(2)) if m.group(2) else 1)
    return v, len(m.group(1).split('.')[1]) if '.' in m.group(1) else 0


def _at(o, path):
    for k in path:
        o = o[k]
    return o


def reference_value(ref, project, results):
    from ..core import library as LB
    from ..core import metrics as M
    if ref[0] == 'metric':
        M.load_metrics()
        return M.read(M.METRICS[ref[2]], results[ref[1]])['value']
    scale = 1.0
    if ref[0] == 'scaled':
        scale, ref = float(ref[1]), ['path'] + list(ref[2:])
    kind, root, path = ref[0], ref[1], ref[2:]
    if root == 'project':
        src = project.to_dict()
    elif root == 'library':
        src = json.loads(_read(LB.get(path[0]).path))
        path = path[1:]
    elif root == 'results':
        src = results
    elif root == 'defaults':
        src = {'resolution': list(LB.DEFAULT_RESOLUTION), 'hfov_deg': LB.DEFAULT_HFOV_DEG,
               'vfov_deg': LB.DEFAULT_VFOV_DEG}
    else:
        raise KeyError(root)
    v = _at(src, path)
    if kind == 'enabled':
        return sum(1 for s in v if s.get('enabled', True))
    return float(v) * scale


@check('publication.table_numbers', 'publication', 'Paper kit: table numbers equal the results',
       'number cells that differ from their source value by more than half a unit of the last digit shown (or '
       '5e-4 of the value), plus table rows with numbers and no source',
       'core.metrics.read on each result; the project, library files and results read here', 0, 'cells',
       refs=('booktabs_fear2020', 'siunitx_wright2026'), cost='study')
def _tables(ctx):
    bad, cells, rows = [], 0, 0
    for si in (False, True):
        folder, res, p = _kit(ctx, si)
        for f in _tex_files(os.path.join(folder, 'tables')):
            for line in _read(f).split('\n'):
                body = re.sub(r'(?<!\\)%.*$', '', line)
                found_cells = _CELL.findall(body)
                m = re.search(r'%\s*dv-src:\s*(\[.*\])\s*$', line)
                if not found_cells:
                    continue
                if not m:
                    bad.append(f'{os.path.basename(f)}: a row without sources')
                    continue
                refs = json.loads(m.group(1))
                rows += 1
                if len(refs) != len(found_cells):
                    bad.append(f'{os.path.basename(f)}: {len(found_cells)} cells, {len(refs)} sources')
                    continue
                for c, ref in zip(found_cells, refs):
                    cells += 1
                    v, d = _value(c)
                    try:
                        want = reference_value(ref, p, res)
                    except (KeyError, IndexError, TypeError, AttributeError) as e:
                        bad.append(f'{c}: no source ({e})')
                        continue
                    if want is None or abs(float(want) - v) > max(0.5 * 10 ** (-d), 5e-4 * abs(v)) * (1 + 1e-9) + 1e-12:
                        bad.append(f'{c} against {want}')
    return Outcome(len(bad), {'cells': cells, 'rows': rows, 'differ': bad[:10]}, ok=cells >= 20)
