"""BibTeX of a project's sources (core/cite.py) and --cite: the file parses (a small parser written here), keys are
unique, and every source the project depends on is there."""
import json
import os
import re
import subprocess
import sys

import pytest

from dexvantage.core import cite as C, library as L
from dexvantage.core.datasets.hub import hub_ref
from dexvantage.core.robot.samples import sample_project
from dexvantage.validation.references import REFERENCES

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def parse(text: str) -> list:
    """[(type, key, {field: value})]: entries found by '@type{key,' and their body by brace counting; fields split at
    top-level commas; an escaped brace (\\{) does not count. Raises on anything malformed."""
    out = []
    pos = 0
    while (i := text.find('@', pos)) >= 0:
        m = re.compile(r'@([A-Za-z]+)\{([^,\s{}]+),').match(text, i)
        assert m, f'entry head at {i}: {text[i:i + 40]!r}'
        k, depth = m.end(), 1
        while depth:
            c = text[k]
            if c == '\\':
                k += 2
                continue
            depth += {'{': 1, '}': -1}.get(c, 0)
            k += 1
        body, fields = text[m.end():k - 1], {}
        j = 0
        while True:
            fm = re.compile(r'\s*([A-Za-z_]+)\s*=\s*\{').match(body, j)
            if not fm:
                assert not body[j:].strip(), f'junk in {m.group(2)}: {body[j:j + 40]!r}'
                break
            s, d = fm.end(), 1
            e = s
            while d:
                c = body[e]
                if c == '\\':
                    e += 2
                    continue
                d += {'{': 1, '}': -1}.get(c, 0)
                e += 1
            fields[fm.group(1).lower()] = body[s:e - 1]
            j = e
            assert body[j] == ',', f'no comma after {fm.group(1)} in {m.group(2)}'
            j += 1
        out.append((m.group(1).lower(), m.group(2), fields))
        pos = k
    return out


@pytest.fixture(scope='module')
def ur5e():
    p = sample_project('ur5e')
    p.data['datasets'].append({'uid': 'd1', 'name': 'Pick and place', 'path': hub_ref('lerobot/svla_so100_pickplace'),
                               'format': 'lerobot', 'version': 'v2.1', 'episodes': None})
    p.data['datasets'].append({'uid': 'd2', 'name': 'Local', 'path': 'recordings/run1', 'format': 'lerobot',
                               'version': 'v2.1', 'episodes': None})
    return p


def test_bibtex_parses_with_unique_keys(ur5e):
    text = C.bibtex(ur5e)
    ents = parse(text)
    keys = [k for _t, k, _f in ents]
    assert len(ents) == len(re.findall(r'^@', text, re.M)) == len(C.entries(ur5e))
    assert len({k.lower() for k in keys}) == len(keys)
    assert all(re.fullmatch(r'[A-Za-z][A-Za-z0-9_:.-]*', k) for k in keys)
    for _t, k, f in ents:
        assert 'title' in f, k
        assert f['title'].count('{') == f['title'].count('}')


def test_every_source_is_cited(ur5e):
    ents = parse(C.bibtex(ur5e))
    by_key = {k: f for _t, k, f in ents}
    urls = {f.get('url') for f in by_key.values()}
    ids = {s.model_id for v in ur5e.variants for s in v.sensors if s.model_id} | {x.library_id for x in ur5e.policies}
    for mid in ids:
        e = L.get(mid)
        with open(e.path, encoding='utf-8') as fh:
            for s in json.load(fh)['sources']:
                assert s['url'] in urls, (mid, s['url'])
                hit = next(f for f in by_key.values() if f.get('url') == s['url'])
                assert hit['urldate'] == s['read']
    sw = by_key['dexvantage']
    assert sw['url'] == 'https://github.com/Sherin-SEF-AI/DexVantage' and 'Sherin Joseph Roy' in sw['author']
    m = [f for f in by_key.values() if 'mujoco_menagerie/tree/' in (f.get('url') or '')]
    assert {u['url'].rsplit('/', 1)[1] for u in m} == {'universal_robots_ur5e', 'robotiq_2f85'}
    assert all('/tree/98a70c569c4c/' in u['url'] for u in m)
    assert 'menagerie2022github' in by_key and by_key['menagerie2022github']['year'] == '2022'
    hub = [f for f in by_key.values() if 'huggingface.co/datasets/' in (f.get('url') or '')]
    assert [f['url'] for f in hub] == ['https://huggingface.co/datasets/lerobot/svla_so100_pickplace']
    assert 'recordings' not in C.bibtex(ur5e)                 # local paths are not citations


def test_references_scopes():
    all_keys = {k for _t, k, _f in parse(C.bibtex(None, 'all'))}
    assert {C._key(k) for k in REFERENCES} <= all_keys
    checks = parse(C.bibtex(None, 'checks'))
    assert 2 < len(checks) <= len(REFERENCES) + 1
    assert [k for _t, k, _f in parse(C.bibtex(None, 'none'))] == ['dexvantage']
    with pytest.raises(ValueError):
        C.entries(None, 'some')


def test_reference_entry_reads_a_paper_and_keeps_others_whole():
    e = C.reference_entry('moller1997', REFERENCES['moller1997'])
    assert e.fields['author'] == 'T. Möller and B. Trumbore'
    assert e.fields['title'] == '{Fast, minimum storage ray-triangle intersection}'
    assert e.fields['year'] == '1997'
    e = C.reference_entry('pinocchio2019', REFERENCES['pinocchio2019'])
    assert e.fields['author'].endswith(' and others')
    e = C.reference_entry('ur_dh', REFERENCES['ur_dh'])
    assert e.fields['author'] == '{Universal Robots}' and e.fields['urldate'] == '2026-11-06'
    assert e.fields['url'].startswith('https://www.universal-robots.com/')
    e = C.reference_entry('json_schema', REFERENCES['json_schema'])
    assert 'author' not in e.fields and e.fields['url'] == 'https://json-schema.org/draft/2020-12'


def test_escaping():
    assert C.escape('a_b & 50% {x} #1 ~ ^ \\') == r'a\_b \& 50\% \{x\} \#1 \textasciitilde{} \textasciicircum{} ' \
                                                 r'\textbackslash{}'


def test_batch_cite(tmp_path):
    out = tmp_path / 'p.bib'
    r = subprocess.run([sys.executable, '-m', 'dexvantage', '--sample', 'so100', '--cite', str(out),
                        '--cite-references', 'none'], capture_output=True, text=True, cwd=ROOT, timeout=120,
                       env=dict(os.environ, QT_QPA_PLATFORM='offscreen'))
    assert r.returncode == 0, r.stderr
    ents = parse(out.read_text(encoding='utf-8'))
    assert ents[0][1] == 'dexvantage' and any(k == 'menagerie-trs_so_arm100' for _t, k, _f in ents)
    assert str(tmp_path) not in out.read_text(encoding='utf-8')
