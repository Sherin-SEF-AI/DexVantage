"""The website (spec §15): every page of site/ is well-formed HTML (html.parser), every internal link and image names a
file of the site and an id in it, the images stay under 400 KB, nothing is loaded from another site, the generated
pages build from the docs, the validation page builds from a report, and the measured figures that the README and the
home page quote equal the validation report's. scripts/build_site.py builds and checks the site. No network."""
import json
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(ROOT, 'site')
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import build_site  # noqa: E402

REPORT = os.path.join(SITE, 'validation', 'validation.json')
CHECK_ID = re.compile(r'`([a-z0-9_]+\.[a-z0-9_]+)`')


def _report_errors() -> dict:
    with open(REPORT, encoding='utf-8') as f:
        rep = json.load(f)
    return {c['id']: c['error'] for m in rep['models'] for c in m['checks']}


# ------------------------------------------------------------------------------------------------ the committed site

def test_site_has_its_pages():
    for page in ('index.html', 'docs.html', 'guide.html', 'methods.html', 'validation.html'):
        assert os.path.isfile(os.path.join(SITE, page)), page
    assert os.listdir(os.path.join(SITE, 'models')) and os.listdir(os.path.join(SITE, 'guide'))


def test_every_page_is_well_formed_html():
    assert build_site.check_wellformed(SITE) == []


def test_every_internal_link_resolves():
    assert build_site.check_links(SITE) == []


def test_images_stay_under_400_kb():
    assert build_site.check_media(SITE) == []
    media = os.listdir(os.path.join(SITE, 'media'))
    assert media and all(f.endswith('.jpg') for f in media)


def test_nothing_is_loaded_from_another_site():
    """Fonts, styles and scripts are self-hosted; no tracker, no web font service, no embedded player."""
    bad = []
    for root, _dirs, files in os.walk(SITE):
        for f in files:
            if not f.endswith(('.html', '.css', '.js')):
                continue
            text = open(os.path.join(root, f), encoding='utf-8').read()
            rel = os.path.relpath(os.path.join(root, f), SITE)
            rxs = [r'<script[^>]+src="(https?:)?//', r'<link[^>]+rel="stylesheet"[^>]+href="(https?:)?//',
                   r'<(img|video|source|iframe|embed|object)[^>]+(src|data)="(https?:)?//', r'<iframe',
                   r'@import', r'url\(\s*["\']?(https?:)?//']
            if not f.endswith('.html'):      # in scripts and styles any of these is a request; in pages, a cited URL
                rxs.append(r'googleapis|googletagmanager|google-analytics|gtag\(|plausible|matomo|youtube|https?:')
            for rx in rxs:
                if re.search(rx, text, re.I):
                    bad.append(f'{rel}: {rx}')
            for script in re.findall(r'<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>', text, re.S | re.I):
                if re.search(r'https?:|fetch\(|XMLHttpRequest|sendBeacon', script):
                    bad.append(f'{rel}: an inline script reaches the network')
    assert not bad, bad
    css = open(os.path.join(SITE, 'assets', 'css', 'site.css'), encoding='utf-8').read()
    for font in re.findall(r'url\("\.\./fonts/([^"]+)"\)', css):
        assert os.path.isfile(os.path.join(SITE, 'assets', 'fonts', font)), font
    assert os.path.isfile(os.path.join(SITE, 'assets', 'fonts', 'OFL.txt'))     # the fonts' licence travels with them


def test_checker_finds_broken_links_and_bad_nesting(tmp_path):
    (tmp_path / 'a.html').write_text(
        '<!doctype html><html><body><div><p id="x">x</div><a href="b.html">b</a><a href="#nope">n</a>'
        '<a href="a.html#x">ok</a><img src="m.jpg" alt=""><span id="x"></span></body></html>', encoding='utf-8')
    (tmp_path / 'c.html').write_text('<html><body><a href="../outside.html">o</a></body></html>', encoding='utf-8')
    wf = build_site.check_wellformed(str(tmp_path))
    assert any('</div> closes <p>' in p for p in wf)
    assert any('id "x" is used twice' in p for p in wf)
    assert any('c.html: no <!doctype html>' in p for p in wf)
    links = build_site.check_links(str(tmp_path))
    assert any('"b.html" names a file that is not there' in p for p in links)
    assert any('"#nope" names an id' in p for p in links)
    assert any('"m.jpg"' in p for p in links)
    assert any('leaves the site' in p for p in links)
    assert not any('a.html#x' in p for p in links)


# ------------------------------------------------------------------------------------------------ building

def test_pages_build_from_the_docs(tmp_path):
    pytest.importorskip('markdown_it', reason='the site extra (markdown-it-py) is not installed')
    warnings = build_site.build_pages(str(tmp_path))
    assert isinstance(warnings, list)
    build_site.build_assets(str(tmp_path))
    for page in ('docs.html', 'guide.html', 'methods.html'):
        assert (tmp_path / page).is_file()
    n_models = len([f for f in os.listdir(os.path.join(ROOT, 'docs', 'models')) if f.endswith('.md')])
    assert len(list((tmp_path / 'models').glob('*.html'))) == n_models
    assert build_site.check_wellformed(str(tmp_path)) == []
    # links into what this temporary build does not write (the home page, the validation page, the hand-written
    # stylesheet and script) are the only misses
    missing = [p for p in build_site.check_links(str(tmp_path))
               if not re.search(r'"(\.\./)?((index|validation)\.html(#[^"]*)?|assets/(css|js)/site\.(css|js))"', p)]
    assert missing == [], missing[:10]
    docs = (tmp_path / 'docs.html').read_text(encoding='utf-8')
    assert 'id="frames"' in docs and 'href="models/robots.html' in docs
    assert not re.search(r'href="(?!https?:)[^"]*\.md[#"]', docs)      # no link to a Markdown file is left relative


def test_markdown_rules(tmp_path, monkeypatch):
    """{#id} headings, raw HTML escaped, links between docs become links between pages, a missing heading is dropped
    with a note, other repository files go to GitHub, a file name as link text becomes the page's title."""
    pytest.importorskip('markdown_it', reason='the site extra (markdown-it-py) is not installed')
    root = tmp_path / 'repo'
    (root / 'docs' / 'models').mkdir(parents=True)
    (root / 'docs' / 'guide').mkdir(parents=True)
    (root / 'LICENSE').write_text('x', encoding='utf-8')
    (root / 'docs' / 'overview.md').write_text(
        '# Overview\n\n## Frames and units {#frames}\n\nA `<b>` and <name>.runs/ stay text. See '
        '[models/a.md](models/a.md), [the limits](models/a.md#known-limits), [gone](models/a.md#nothing), '
        '[licence](../LICENSE), [web](https://example.org) and [missing](nothere.md).\n\n'
        '| a | b |\n|---|---|\n| 1 | 2 |\n\n    p = B · q        [Shoemake 1985]\n', encoding='utf-8')
    (root / 'docs' / 'user-guide.md').write_text('# Guide\n\n## Start\n\n<!-- example: tour-so100 -->\n'
                                                 'Read [the frames](overview.md#frames).\n', encoding='utf-8')
    (root / 'docs' / 'models' / 'a.md').write_text('# Model A\n\nFirst paragraph.\n\n## Known limits\n\nNone.\n',
                                                   encoding='utf-8')
    (root / 'docs' / 'guide' / 'a.md').write_text('# Example A\n\nSee [model](../models/a.md).\n\n'
                                                  '![The tour](../media/tour-a.png) ![gone](../media/none.png)\n',
                                                  encoding='utf-8')
    (root / 'docs' / 'media').mkdir()
    (root / 'docs' / 'media' / 'tour-a.png').write_bytes(b'\x89PNG fake')
    monkeypatch.setattr(build_site, 'ROOT', str(root))
    monkeypatch.setattr(build_site, 'DOCS', str(root / 'docs'))
    site = tmp_path / 'site'
    warnings = build_site.build_pages(str(site))
    html = (site / 'docs.html').read_text(encoding='utf-8')
    assert '<h2 id="frames">Frames and units</h2>' in html
    assert '&lt;b&gt;' in html and '&lt;name&gt;.runs/' in html and '<name>' not in html
    assert '<a href="models/a.html">Model A</a>' in html
    assert 'href="models/a.html#known-limits"' in html
    assert 'href="models/a.html">gone</a>' in html
    assert f'href="{build_site.BLOB_URL}LICENSE"' in html
    assert 'href="https://example.org"' in html
    assert '>missing<' not in html and 'missing' in html          # shown as text
    assert '<table>' in html and '<pre><code>p = B · q' in html     # equations as written, in a code block
    assert any('models/a.md#nothing' in w for w in warnings)
    assert any('nothere.md' in w for w in warnings)
    guide = (site / 'guide.html').read_text(encoding='utf-8')
    assert 'example: tour-so100' not in guide and '&lt;!--' not in guide     # the source-only marker is dropped
    assert 'href="docs.html#frames"' in guide and 'id="worked-examples"' in guide and 'href="guide/a.html"' in guide
    page_a = (site / 'guide' / 'a.html').read_text(encoding='utf-8')
    assert 'href="../models/a.html"' in page_a
    # a repository image is copied into the site and loaded from there (a page loads nothing from another site)
    assert '<img src="../docs-media/tour-a.png" alt="The tour"' in page_a and 'blob/main/docs/media' not in page_a
    assert (site / 'docs-media' / 'tour-a.png').read_bytes() == b'\x89PNG fake'
    assert 'none.png' not in page_a and any('none.png' in w for w in warnings)
    assert 'href="models/a.html"' in (site / 'methods.html').read_text(encoding='utf-8')
    assert build_site.check_wellformed(str(site)) == []


def test_validation_page_from_a_report(tmp_path):
    rep = {'schema': 'dexvantage.validation/1', 'created_utc': '2026-01-02T03:04:05Z', 'seconds': 12.5,
           'environment': {'dexvantage': '0.1.0', 'platform': 'Linux x86_64'}, 'settings': {'seed': 1},
           'frame': 'world', 'limits': 'Checks the cases it builds.',
           'summary': {'checks': 2, 'passed': 1, 'failed': ['m.b'], 'ok': False},
           'references': {'r1': 'A reference <with brackets>.'},
           'models': [{'id': 'm', 'title': 'Model M', 'methods_anchor': 'nowhere', 'modules': 'dexvantage.m',
                       'checks': [
                           {'id': 'm.a', 'title': 'A', 'quantity': 'q', 'reference': 'ref', 'references': ['r1'],
                            'error': 1.5e-15, 'tolerance': 1e-12, 'unit': 'm', 'passed': True, 'note': '',
                            'figure': {'kind': 'xy'}},
                           {'id': 'm.b', 'title': 'B', 'quantity': 'q', 'reference': 'ref', 'references': [],
                            'error': 0.25, 'tolerance': 0.1, 'unit': '(unitless)', 'passed': False, 'note': 'n',
                            'figure': None}]}]}
    src = tmp_path / 'report'
    (src / 'figures').mkdir(parents=True)
    (src / 'validation.json').write_text(json.dumps(rep), encoding='utf-8')
    (src / 'validation.csv').write_text('id\n', encoding='utf-8')
    (src / 'figures' / 'm.a.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg"/>', encoding='utf-8')
    site = tmp_path / 'site'
    site.mkdir()
    s = build_site.build_validation(str(src), str(site))
    assert s == {'checks': 2, 'passed': 1, 'models': 1, 'date': '2026-01-02'}
    page = (site / 'validation.html').read_text(encoding='utf-8')
    assert '1 of 2' in page and 'Failed: m.b' in page and '1.5 × 10⁻¹⁵' in page
    assert 'id="c-m.a"' in page and 'href="validation/figures/m.a.svg"' in page
    assert 'A reference &lt;with brackets&gt;.' in page
    assert 'href="docs.html"' in page                       # an anchor docs.html does not have is not linked
    assert (site / 'validation' / 'validation.json').is_file()
    assert build_site.check_wellformed(str(site)) == []
    assert [p for p in build_site.check_links(str(site)) if 'index.html' not in p and 'guide.html' not in p
            and 'docs.html' not in p and 'methods.html' not in p and 'assets/' not in p] == []


def test_validation_refusals_are_plain_sentences(tmp_path):
    with pytest.raises(build_site.SiteError, match='There is no validation report'):
        build_site.build_validation(str(tmp_path / 'none'), str(tmp_path))
    (tmp_path / 'validation.json').write_text('{"schema": "other"}', encoding='utf-8')
    with pytest.raises(build_site.SiteError, match='is not a DexVantage validation report'):
        build_site.build_validation(str(tmp_path), str(tmp_path))
    (tmp_path / 'validation.json').write_text('{', encoding='utf-8')
    with pytest.raises(build_site.SiteError, match='could not be read'):
        build_site.build_validation(str(tmp_path), str(tmp_path))
    with pytest.raises(build_site.SiteError, match='is not in'):
        build_site.build_media(str(tmp_path / 'no-screens'), str(tmp_path / 's'), str(tmp_path / 'd'))


def test_check_mode_needs_no_markdown_package(capsys):
    assert build_site.main(['--check']) == 0
    assert 'no problems' in capsys.readouterr().out


@pytest.mark.parametrize('x, text', [(0, '0'), (1.1102230246251565e-15, '1.11 × 10⁻¹⁵'), (0.001835, '0.001835'),
                                     (385.57, '385.6'), (16.87, '16.87'), (12345.6, '12 346'), (-2.5e-7, '−2.5 × 10⁻⁷'),
                                     (1e-12, '1 × 10⁻¹²'), (None, '—'), (float('nan'), '—')])
def test_numbers_are_written_as_the_app_writes_them(x, text):
    assert build_site.fmt_number(x) == text


# ------------------------------------------------------------------------------------------------ quoted figures

def _rows_citing_checks(path):
    rows = []
    for line in open(path, encoding='utf-8'):
        cells = [c for c in line.strip().split('|') if c.strip()] if line.startswith('|') else []
        where = cells[-1] if cells else ''                    # the "Where measured" cell names the checks
        ids = CHECK_ID.findall(where) if re.search(r'\bchecks? `', where) else []
        if ids:
            rows.append((line, ids))
    return rows


def test_readme_figures_equal_the_validation_report():
    errors = _report_errors()
    rows = _rows_citing_checks(os.path.join(ROOT, 'README.md'))
    assert len(rows) >= 30
    bad = []
    for line, ids in rows:
        for cid in ids:
            if cid not in errors:
                bad.append(f'{cid}: not in the report')
            elif build_site.fmt_number(errors[cid]) not in line:
                bad.append(f'{cid}: the report says {build_site.fmt_number(errors[cid])}')
    assert not bad, bad


def test_home_page_figures_equal_the_validation_report():
    errors = _report_errors()
    page = open(os.path.join(SITE, 'index.html'), encoding='utf-8').read()
    rows = re.findall(r'<tr><td>.*?</tr>', page)
    cited = 0
    for row in rows:
        m = re.search(r'validation\.html#c-([a-z0-9_.]+)', row)
        if not m:
            continue
        cited += 1
        cid, num = m.group(1), re.search(r'<td class="num">([^<]*)</td>', row).group(1)
        if cid == 'groundtruth.ur5e_contacts':          # quoted as the agreement, 1 − error
            assert num == f'{100 * (1 - errors[cid]):.1f} % agree', row
        else:
            assert num.startswith(build_site.fmt_number(errors[cid])), row
    assert cited >= 10


def test_readme_names_the_bundles_and_the_site():
    text = open(os.path.join(ROOT, 'README.md'), encoding='utf-8').read()
    for name in ('-windows-x64-setup.exe', '-macos-arm64.zip', '-linux-x86_64.tar.gz', '-linux-arm64.tar.gz'):
        assert f'DexVantage-<version>{name}' in text, name
    assert 'once released' in text and 'GPL-3.0-or-later' in text and 'CITATION.cff' in text
    for img in re.findall(r'\]\((docs/media/[^)]+)\)', text):
        assert os.path.isfile(os.path.join(ROOT, img)), img
    for link in re.findall(r'\]\((?!https?:|#)([^)#]+)', text):
        assert os.path.exists(os.path.join(ROOT, link)), link


def test_pages_workflow_is_pinned_and_checks_the_site():
    text = open(os.path.join(ROOT, '.github', 'workflows', 'pages.yml'), encoding='utf-8').read()
    uses = re.findall(r'uses:\s*(\S+)(.*)', text)
    assert {u.split('@')[0] for u, _ in uses} == {'actions/checkout', 'actions/configure-pages',
                                                  'actions/upload-pages-artifact', 'actions/deploy-pages'}
    for ref, comment in uses:
        assert re.fullmatch(r'[\w./-]+@[0-9a-f]{40}', ref), ref
        assert re.search(r'#\s*v\d+\.\d+\.\d+', comment), ref
    assert 'scripts/build_site.py --check' in text and 'path: site' in text
