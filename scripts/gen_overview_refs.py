"""Keep docs/overview.md's references in step with the validation checks.

For every model of `validation/suite.py` MODELS, the overview section at its anchor ends with one line
`**References (checks):** key, key, …` naming every reference its checks cite (`@check(..., refs=...)`), and the
overview ends with `## References {#references}`: each key cited anywhere in the checks with its entry from
`validation/references.py`. Only those lines are written; the rest of the overview is the authors'.

    python scripts/gen_overview_refs.py            # rewrite those lines in docs/overview.md
    python scripts/gen_overview_refs.py --check    # exit 1 when they are out of date

tests/test_docs_reference.py runs the check.
"""
from __future__ import annotations

import argparse
import collections
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
OVERVIEW = os.path.join(ROOT, 'docs', 'overview.md')
LINE = '**References (checks):** '
REF_HEADING = '## References {#references}'


def model_refs() -> tuple[dict, dict]:
    """({model id: sorted reference keys}, REFERENCES) from the registered checks."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from dexvantage.validation import references as R
    from dexvantage.validation import suite as S
    S.load_checks()
    by = collections.defaultdict(set)
    for c in S.CHECKS:
        by[c.model].update(c.refs)
    return {m[0]: sorted(by.get(m[0], ())) for m in S.MODELS}, R.REFERENCES


def section_span(text: str, anchor: str) -> tuple[int, int] | None:
    m = re.search(r'^## [^\n]*\{#' + re.escape(anchor) + r'\}[^\n]*$', text, re.M)
    if not m:
        return None
    n = re.search(r'^## ', text[m.end():], re.M)
    return m.start(), (m.end() + n.start()) if n else len(text)


def render(text: str) -> str:
    from dexvantage.validation import suite as S
    refs, entries = model_refs()
    for mid, _title, _mods, anchor in S.MODELS:
        span = section_span(text, anchor)
        if span is None:
            continue                                  # the test names a missing section; nothing to write into
        a, b = span
        body = '\n'.join(ln for ln in text[a:b].rstrip('\n').split('\n') if not ln.startswith(LINE)).rstrip('\n')
        keys = refs[mid]
        line = LINE + (', '.join(f'`{k}`' for k in keys) if keys else 'none (its checks cite no outside source)') + \
            ' ([full entries](#references)).'
        text = text[:a] + body + '\n\n' + line + '\n\n' + text[b:].lstrip('\n')
    cited = sorted({k for ks in refs.values() for k in ks})
    block = [REF_HEADING, '',
             'Every source the validation checks cite, by key (`dexvantage/validation/references.py`; the model '
             'documents cite more).', '']
    block += [f'- `{k}`: {entries[k]}' for k in cited]
    i = text.find(REF_HEADING)
    text = (text[:i].rstrip('\n') if i >= 0 else text.rstrip('\n')) + '\n\n' + '\n'.join(block) + '\n'
    return re.sub(r'\n{3,}', '\n\n', text)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Write the checks' references into docs/overview.md.")
    p.add_argument('--check', action='store_true', help='exit 1 when the references are out of date')
    args = p.parse_args(argv)
    have = open(OVERVIEW, encoding='utf-8').read()
    text = render(have)
    if args.check:
        if have != text:
            print('docs/overview.md references are out of date: run python scripts/gen_overview_refs.py')
            return 1
        print('docs/overview.md references are up to date.')
        return 0
    with open(OVERVIEW, 'w', encoding='utf-8') as fh:
        fh.write(text)
    print('wrote the references of docs/overview.md')
    return 0


if __name__ == '__main__':
    sys.exit(main())
