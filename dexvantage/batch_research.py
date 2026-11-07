"""Batch mode's research tools, added to batch.py's parser: project diff and merge, the Git-friendly folder form,
citations and publication figures.

    dexvantage A.dexv --diff B.dexv [OUT.json]        what changed from A to B: a table on stdout, the JSON (schema
                                                      dexvantage.diff/1, with provenance) to OUT.json; exit 0 when
                                                      the projects are the same, 1 when they differ (as diff does)
        --diff-html OUT.html                          also the table as a standalone HTML page
        --diff-metrics ID,ID,...                      also run these metrics' studies on both and give the deltas
    dexvantage OURS.dexv --merge BASE.dexv THEIRS.dexv OUT.dexv
                                                      merge two edits of BASE; exit 1 (nothing written) when they
                                                      conflict, each conflict said
    dexvantage P.dexv --unpack P.dexv.d               the project as an unpacked folder (core/projfolder.py)
    dexvantage P.dexv.d --pack P.dexv                 an unpacked folder as a .dexv file
    dexvantage --textconv FILE                        a stable text of a .dexv file or folder for git diff
    dexvantage P.dexv --cite OUT.bib [--cite-references checks|all|none]
                                                      BibTeX of every source the project depends on (core/cite.py)
    dexvantage --figure SPEC.json OUT.svg|OUT.pdf [--figure-width single|double|MM] [--figure-palette tol|chart]
                                                      a publication figure from a figure spec (core/figures.py; the
                                                      pages' "Export figure…" writes such specs into their SVGs)

Exit codes: 0 done, 1 a diff that found differences or a merge that conflicts, 2 an error (a file that cannot be read,
a bad option). Errors are one sentence on stderr.
"""
from __future__ import annotations

import os
import sys

EXIT_OK, EXIT_FAIL, EXIT_ERROR = 0, 1, 2


def add_arguments(g):
    from .core import provenance as PV
    g.add_argument('--diff', nargs='+', metavar=('OTHER.dexv', 'OUT.json'),
                   help='compare the project with another (a .dexv file or unpacked folder): print the changes, '
                        'write them as JSON to OUT.json; exit 0 same, 1 different')
    g.add_argument('--diff-html', metavar='OUT.html', help='with --diff: also write the changes as an HTML page')
    g.add_argument('--diff-metrics', metavar='ID,ID,...', help='with --diff: run these metrics on both projects and '
                   'give their deltas (core/metric_defs.py lists the ids)')
    g.add_argument('--merge', nargs=3, metavar=('BASE.dexv', 'THEIRS.dexv', 'OUT.dexv'),
                   help='three-way merge: the project (ours) and THEIRS, both edits of BASE, written to OUT (a .dexv '
                        'file or a .dexv.d folder); exit 1 and write nothing when they conflict')
    g.add_argument('--unpack', metavar='OUT.dexv.d', help='write the project as an unpacked, Git-friendly folder')
    g.add_argument('--pack', metavar='OUT.dexv', help='write the project (for example an unpacked folder) as a .dexv '
                   'file')
    g.add_argument('--textconv', metavar='FILE', help='print a stable text of a .dexv file or folder (a git diff '
                   'textconv driver: docs/guide/versioning.md)')
    g.add_argument('--cite', metavar='OUT.bib', help='write BibTeX for every source the project depends on and for '
                   'DexVantage ("-" for stdout)')
    g.add_argument('--cite-references', choices=('checks', 'all', 'none'), help='with --cite: the validation '
                   'references to include (default checks: those the validation checks cite)')
    g.add_argument('--figure', nargs=2, metavar=('SPEC.json', 'OUT.svg'), help='write a publication figure (SVG or '
                   'PDF by the extension) from a figure spec (core/figures.py)')
    g.add_argument('--figure-width', metavar='single|double|MM', help='with --figure: one column (88.9 mm, default), '
                   'two columns (182 mm) or a width in mm')
    g.add_argument('--figure-palette', choices=('tol', 'chart'), help='with --figure: tol (colour-blind safe, '
                   'default) or chart (the colours in the spec)')
    PV.INPUT_ARGS.setdefault('diff', 'first')
    PV.INPUT_ARGS.setdefault('merge', 'all')
    PV.INPUT_ARGS.setdefault('figure', 'first')


_ALL = ('diff', 'diff_html', 'diff_metrics', 'merge', 'unpack', 'pack', 'textconv', 'cite', 'cite_references',
        'figure', 'figure_width', 'figure_palette')


def wanted(args) -> bool:
    return any(getattr(args, k, None) for k in _ALL)


def _err(text) -> int:
    print(f'dexvantage: {text}', file=sys.stderr)
    return EXIT_ERROR


def _pairs_ok(args) -> str | None:
    if (getattr(args, 'diff_html', None) or getattr(args, 'diff_metrics', None)) and not getattr(args, 'diff', None):
        return '--diff-html and --diff-metrics go with --diff OTHER.dexv'
    if getattr(args, 'diff', None) and len(args.diff) > 2:
        return '--diff takes the other project and, optionally, the JSON file to write'
    if getattr(args, 'cite_references', None) and not getattr(args, 'cite', None):
        return '--cite-references goes with --cite OUT.bib'
    if (getattr(args, 'figure_width', None) or getattr(args, 'figure_palette', None)) and \
            not getattr(args, 'figure', None):
        return '--figure-width and --figure-palette go with --figure SPEC.json OUT.svg'
    return None


def textconv(args) -> int:
    """--textconv FILE: the text only, on stdout (git reads it)."""
    from .core import projfolder as PF
    from .core.project import ProjectFormatError
    try:
        sys.stdout.write(PF.textconv(args.textconv))
        sys.stdout.flush()
    except BrokenPipeError:                             # the reader stopped early (a pager): not an error
        try:
            sys.stdout = open(os.devnull, 'w')
        except OSError:
            pass
        return EXIT_OK
    except ProjectFormatError as e:
        return _err(str(e))
    except OSError as e:
        return _err(f'could not read {os.path.basename(args.textconv)}: {e.strerror or e}.')
    return EXIT_OK


def run(args, project) -> int:
    """Every research option given (after batch.py's own); the worst exit code."""
    bad = _pairs_ok(args)
    if bad:
        return _err(bad)
    code = EXIT_OK
    steps = (('diff', diff), ('merge', merge), ('unpack', unpack), ('pack', pack), ('cite', cite), ('figure', figure))
    for k, fn in steps:
        if getattr(args, k, None):
            code = max(code, fn(args, project))
            if code == EXIT_ERROR:
                break
    return code


def _write_text(text: str, dest: str) -> int | None:
    try:
        with open(dest + '.part', 'w', encoding='utf-8') as fh:
            fh.write(text)
        os.replace(dest + '.part', dest)
    except OSError as e:
        return _err(f'could not write {dest}: {e.strerror or e}. Choose a folder that exists and that you can write '
                    'to.')
    return None


def diff(args, project) -> int:
    import json

    from .core import jobs, projdiff as D, provenance as PV
    from .core.project import Project, ProjectFormatError
    if project is None:
        return _err('--diff needs a project file to compare (dexvantage A.dexv --diff B.dexv)')
    other_path = args.diff[0]
    try:
        other = Project.load(other_path)
    except ProjectFormatError as e:
        return _err(str(e))
    try:
        doc = D.diff(project, other)
        if getattr(args, 'diff_metrics', None):
            ids = [x.strip() for x in args.diff_metrics.split(',') if x.strip()]
            doc['metrics'] = D.evaluate_deltas(project, other, ids)
    except jobs.StudyError as e:
        return _err(str(e))
    sys.stdout.write(D.to_text(doc))
    out = args.diff[1] if len(args.diff) > 1 else None
    if out:
        stamped = PV.stamp(doc, project, output=PV.batch_output(doc, out, option='diff'))
        text = json.dumps(stamped, indent=1, ensure_ascii=False, allow_nan=False) + '\n'
        if out == '-':
            sys.stdout.write(text)
        elif (e := _write_text(text, out)) is not None:
            return e
    if getattr(args, 'diff_html', None):
        if (e := _write_text(D.to_html(doc), args.diff_html)) is not None:
            return e
    return EXIT_OK if doc['identical'] else EXIT_FAIL


def merge(args, project) -> int:
    from .core import jobs, projdiff as D
    from .core.project import ProjectFormatError
    if project is None:
        return _err('--merge needs our project file first (dexvantage OURS.dexv --merge BASE.dexv THEIRS.dexv OUT)')
    base, theirs, out = args.merge
    try:
        res = D.merge3(base, project, theirs)
    except D.MergeConflict as e:
        print(f'dexvantage: {e}', file=sys.stderr)
        return EXIT_FAIL
    except (ProjectFormatError, jobs.StudyError) as e:
        return _err(str(e))
    try:
        res.project.save(out)
    except (OSError, ProjectFormatError) as e:
        return _err(f'could not write {out}: {getattr(e, "strerror", None) or e}.')
    print(f'Merged: {len(res.from_ours)} change{"" if len(res.from_ours) == 1 else "s"} of ours, '
          f'{len(res.from_theirs)} of theirs → {out}')
    for n in res.notes:
        print(f'  · {n}')
    return EXIT_OK


def unpack(args, project) -> int:
    from .core import projfolder as PF
    from .core.project import ProjectFormatError
    if project is None:
        return _err('--unpack needs a project file (dexvantage P.dexv --unpack P.dexv.d)')
    try:
        r = PF.save_unpacked(project, args.unpack)
    except (OSError, ProjectFormatError) as e:
        return _err(f'could not write {args.unpack}: {getattr(e, "strerror", None) or e}.')
    print(f'Unpacked {project.name}: {len(r["written"])} file{"" if len(r["written"]) == 1 else "s"} written, '
          f'{r["unchanged"]} unchanged, {len(r["removed"])} removed → {args.unpack}')
    return EXIT_OK


def pack(args, project) -> int:
    from .core import projfolder as PF
    from .core.project import ProjectFormatError
    if project is None:
        return _err('--pack needs a project (dexvantage P.dexv.d --pack P.dexv)')
    if PF.is_unpacked(args.pack):
        return _err('--pack writes a .dexv file; use --unpack to write a folder')
    try:
        project.save(args.pack)
    except (OSError, ProjectFormatError) as e:
        return _err(f'could not write {args.pack}: {getattr(e, "strerror", None) or e}.')
    print(f'Packed {project.name} → {args.pack}')
    return EXIT_OK


def cite(args, project) -> int:
    from .core import cite as C
    try:
        r = C.write(project, args.cite, getattr(args, 'cite_references', None) or 'checks')
    except OSError as e:
        return _err(f'could not write {args.cite}: {e.strerror or e}.')
    if args.cite != '-':
        print(f'{r["entries"]} BibTeX entries → {args.cite}')
        for mid in C.missing_library(project):
            print(f'  · {mid} is not in the library, so its sources are not cited', file=sys.stderr)
    return EXIT_OK


def figure(args, project) -> int:
    from .core import figures as F
    spec_path, out = args.figure
    width = getattr(args, 'figure_width', None) or 'single'
    if width not in F.WIDTHS_MM:
        try:
            width = float(width)
        except ValueError:
            return _err('--figure-width is single, double or a width in mm')
    try:
        spec = F.read_spec(spec_path)
        r = F.write(spec, out, width=width, palette=getattr(args, 'figure_palette', None) or 'tol',
                    meta={'project': project.name} if project is not None else None)
    except F.FigureError as e:
        return _err(str(e))
    print(f'Figure ({r["format"].upper()}, {r["width_mm"]:g} × {r["height_mm"]:.1f} mm, {r["series"]} '
          f'series) → {out}')
    return EXIT_OK
