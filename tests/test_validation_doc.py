"""docs/validation.md is made by scripts/gen_validation_doc.py from a full validation report of the checks registered
now: the page lists exactly the current check list, and the script refuses a report that does not match it."""
import json
import os
import re
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import gen_validation_doc as G  # noqa: E402

from dexvantage.validation import suite  # noqa: E402
from dexvantage.validation.references import REFERENCES  # noqa: E402

DOC = os.path.join(ROOT, 'docs', 'validation.md')


def fake_report(drop=(), extra=(), fail=(), **settings):
    """A report of the shape dexvantage --validate writes, for every registered check (errors half the tolerance)."""
    specs = [s for s in suite.registered() if s.id not in drop]
    models, used = [], []
    for mid, title, mods, anchor in suite.all_models():
        cs = [{'id': s.id, 'model': s.model, 'title': s.title, 'quantity': s.quantity, 'reference': s.reference,
               'references': list(s.refs), 'error': s.tolerance / 2, 'tolerance': s.tolerance, 'unit': s.unit,
               'passed': s.id not in fail, 'seconds': 0.1, 'note': f'why {s.id}', 'exception': '', 'detail': {},
               'figure': None} for s in specs if s.model == mid]
        if mid == 'groundtruth':
            cs += [{'id': x, 'model': mid, 'title': x, 'quantity': 'q', 'reference': 'r', 'references': [],
                    'error': 0.0, 'tolerance': 1.0, 'unit': 'm', 'passed': True, 'note': '', 'exception': '',
                    'detail': {}} for x in extra]
        for c in cs:
            used += [r for r in c['references'] if r not in used]
        if cs:
            models.append({'id': mid, 'title': title, 'modules': mods, 'methods_anchor': anchor,
                           'passed': sum(c['passed'] for c in cs), 'checks': cs})
    st = {'models': [], 'quick': False, 'seed': 7, 'cheap_only': False, **settings}
    n = sum(len(m['checks']) for m in models)
    return {'schema': 'dexvantage.validation/1', 'created_utc': '2026-11-08T00:00:00Z', 'seconds': 12.0,
            'settings': st, 'environment': {'dexvantage': '0', 'python': '3', 'platform': 'Linux x86_64'},
            'frame': 'world', 'limits': 'limits', 'cancelled': False,
            'summary': {'checks': n, 'passed': n - len(fail), 'failed': list(fail), 'ok': not fail},
            'models': models, 'references': {k: REFERENCES.get(k, '') for k in used}}


def doc_ids(text: str) -> list:
    every = text.split('## Every check, by model', 1)[1].split('## References', 1)[0]
    return re.findall(r'^\| `([^`]+)` \|', every, re.M)


def test_the_page_lists_the_current_check_list():
    assert os.path.exists(DOC), 'docs/validation.md is missing: run scripts/gen_validation_doc.py'
    with open(DOC, encoding='utf-8') as fh:
        text = fh.read()
    assert 'Made by scripts/gen_validation_doc.py' in text
    specs = suite.registered()                     # also the checks this machine leaves out (CI: no PyTorch, no CUDA)
    ids = doc_ids(text)
    assert len(ids) == len(set(ids))
    assert sorted(ids) == sorted(s.id for s in specs), (
        'docs/validation.md does not list the registered checks: run dexvantage --validate out/validation and '
        'python scripts/gen_validation_doc.py out/validation/validation.json')
    for s in specs:
        assert s.reference in text, s.id
        assert s.quantity in text, s.id
    for mid, title, _mods, _a in suite.all_models():
        if any(s.model == mid for s in specs):
            assert f'### {title}' in text
    assert re.search(rf'of {len(specs):,} checks pass'.replace(',', ' '), text)
    assert '## Agreement with MuJoCo renders' in text and '`groundtruth.ur5e_contacts`' in text
    assert not G.PERSONAL.search(text)


def test_the_script_writes_every_check_with_its_reference(tmp_path):
    rep = fake_report(fail=('groundtruth.ur5e_depth',))
    for m in rep['models']:
        for c in m['checks']:
            if c['id'] == 'groundtruth.ur5e_contacts':
                c['detail'] = {'agreement': 0.9275, 'iou': 0.88, 'by_kind': {'home': {'agreement': 0.98, 'n': 512}}}
    src, out = tmp_path / 'validation.json', tmp_path / 'validation.md'
    src.write_text(json.dumps(rep), encoding='utf-8')
    assert G.main([str(src), '--out', str(out)]) == 0
    text = out.read_text(encoding='utf-8')
    assert sorted(doc_ids(text)) == sorted(s.id for s in suite.registered())
    assert 'agreement 92.75 %, IoU 0.88; by case: home 98 % of 512 points' in text.replace('98.00 %', '98 %')
    assert '| `groundtruth.ur5e_depth` |' in text and 'FAIL' in text and '**Failing checks:**' in text
    for k in rep['references']:
        assert f'<a id="ref-{k}"></a>' in text


@pytest.mark.parametrize('kind', ['missing', 'extra', 'quick', 'models', 'junk'])
def test_the_script_refuses_a_report_that_is_not_a_full_run_of_the_current_checks(tmp_path, capsys, kind):
    first = suite.registered()[0].id
    rep = {'missing': lambda: fake_report(drop=(first,)), 'extra': lambda: fake_report(extra=('groundtruth.gone',)),
           'quick': lambda: fake_report(quick=True), 'models': lambda: fake_report(models=['frames']),
           'junk': lambda: {'hello': 1}}[kind]()
    src, out = tmp_path / 'validation.json', tmp_path / 'validation.md'
    src.write_text(json.dumps(rep), encoding='utf-8')
    assert G.main([str(src), '--out', str(out)]) == 1
    err = capsys.readouterr().err
    assert err.startswith('gen_validation_doc: ') and 'Traceback' not in err and len(err.splitlines()) == 1
    assert not out.exists()
    if kind == 'missing':
        assert first in err
    if kind == 'extra':
        assert 'groundtruth.gone' in err


def test_numbers_are_written_from_the_report_as_they_are():
    assert G.num(0) == '0' and G.num(None) == '—' and G.num(float('nan')) == '—'
    assert G.num(1.11e-15) == '1.11 × 10⁻¹⁵' and G.num(0.000885) == '8.85 × 10⁻⁴'
    assert G.num(2314000) == '2 314 000' and G.num(0.0724559) == '0.07246'
    assert G.pct(0.9999942) == '99.999 %' and G.pct(1.0) == '100.00 %'
