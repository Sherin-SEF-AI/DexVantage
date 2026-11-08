"""Build the website's generated pages (spec §15) and check the whole site.

    python scripts/build_site.py                  # docs.html, guide.html, methods.html, models/*.html, guide/*.html
    python scripts/build_site.py --validation     # validation.html from out/validation (a --validate report)
    python scripts/build_site.py --validation DIR # ... from another report folder
    python scripts/build_site.py --media          # screenshots from out/screens into site/media and docs/media
    python scripts/build_site.py --check          # link check and HTML well-formedness over site/ (no build)

The site is static and lives in `site/` (deployed as it is by `.github/workflows/pages.yml`). `index.html` is written
by hand; everything else here is generated from the repository:

- `docs.html` from `docs/overview.md` (every model with its equations, references and limits),
- `guide.html` from `docs/user-guide.md`, followed by the list of worked examples,
- `models/<name>.html` from `docs/models/<name>.md` and `guide/<name>.html` from `docs/guide/<name>.md`,
- `methods.html`, the index of the model documents,
- `validation.html` (with `--validation`) from the `validation.json` that `dexvantage --validate DIR` writes, with its
  CSV, JSON and figures copied under `validation/`.

Markdown is converted with markdown-it-py (CommonMark, plus tables and strikethrough; pinned in the `site` extra of
`pyproject.toml`). Raw HTML in the Markdown is escaped, never passed through, so a `<name>` in the text stays text.
Equations are written in the docs as Unicode text, most in indented code blocks; the site shows them as written, in
IBM Plex Mono. There is no KaTeX and nothing is loaded from another site: fonts, styles and scripts are under
`site/assets`.

Links between the Markdown files become links between the pages; a link to any other file in the repository goes to
that file on GitHub. A link to a heading that does not exist in the target page keeps the page and drops the heading
(the build prints each one so the Markdown can be fixed). `--check` (standard library only) then checks every internal
link and image of every page and that every page is well-formed HTML.

Ported from Vantage `site/` (its stylesheet, script, page frame and navigation); the pages' content is DexVantage's.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import posixpath
import re
import shutil
import sys
from html.parser import HTMLParser

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
SITE = os.path.join(ROOT, 'site')
DOCS = os.path.join(ROOT, 'docs')
REPO_URL = 'https://github.com/Sherin-SEF-AI/DexVantage'
BLOB_URL = REPO_URL + '/blob/main/'
TREE_URL = REPO_URL + '/tree/main/'
RELEASES_URL = REPO_URL + '/releases/latest'
MEDIA_LIMIT = 400 * 1024          # bytes per image under site/media and docs/media (the site's budget)

# Screenshots for the site and the README, made by scripts/screens.py into out/screens (PNG): (source, name, readme)
MEDIA = [
    ('phase3-ur5e-grasp-page.png', 'grasp.jpg', True),
    ('sample-aloha-main.png', 'aloha.jpg', True),
    ('sample-so100-main.png', 'so100.jpg', False),
    ('sample-ur5e-main.png', 'ur5e.jpg', False),
    ('sample-panda-bimanual-main.png', 'panda.jpg', False),
    ('sample-g1-main.png', 'g1.jpg', False),
    ('phase2-ur5e-coverage-page.png', 'coverage.jpg', False),
    ('phase2-ur5e-depth-page.png', 'depth.jpg', False),
    ('phase3-ur5e-vla-page.png', 'vla.jpg', False),
    ('phase3-dexhand-matrix-page.png', 'dexhand.jpg', True),
    ('phase3-ur5e-bandwidth-topology.png', 'bandwidth.jpg', False),
    ('phase3-ur5e-brackets-preview.png', 'brackets.jpg', False),
    ('phase3-ur5e-handeye-plan.png', 'handeye.jpg', False),
    ('p458-ur5e-groundtruth-page.png', 'groundtruth.jpg', True),
    ('p458-so100-teleop-page.png', 'teleop.jpg', False),
    ('p458-ur5e-lighting-page.png', 'lighting.jpg', False),
    ('p458-so100-audit-page.png', 'audit.jpg', True),
    ('p910-ur5e-optimise-pareto.png', 'optimise.jpg', False),
    ('rt2-aloha-campaign-page.png', 'campaign.jpg', True),
    ('rt2-ur5e-statistics-page.png', 'statistics.jpg', False),
    ('rt2-ur5e-sim2real-page.png', 'sim2real.jpg', False),
]

NAV = [  # (label, target page in site/, current-page key)
    ('Features', 'index.html#features', None),
    ('Samples', 'index.html#samples', None),
    ('Guide', 'guide.html', 'guide'),
    ('Docs', 'docs.html', 'docs'),
    ('Methods', 'methods.html', 'methods'),
    ('Validation', 'validation.html', 'validation'),
    ('Download', 'index.html#download', None),
]


class SiteError(Exception):
    """A plain-sentence reason the site cannot be built."""


# ------------------------------------------------------------------------------------------------ Markdown → HTML

def _markdown():
    try:
        from markdown_it import MarkdownIt
    except ImportError:
        raise SiteError('The site build needs markdown-it-py: install the site extra with '
                        '".venv/bin/pip install -e .[site]".') from None
    return MarkdownIt('commonmark', {'html': False}).enable(['table', 'strikethrough'])


def gh_slug(text: str) -> str:
    """GitHub's heading anchor: lower case, punctuation removed (letters, digits, _, - and spaces kept), spaces to -."""
    s = re.sub(r'[^\w\- ]', '', text.strip().lower())
    return s.replace(' ', '-')


def _inline_text(tok) -> str:
    return ''.join(c.content for c in (tok.children or []) if c.type in ('text', 'code_inline', 'html_inline'))


_COMMENT_LINE = re.compile(r'^[ \t]*<!--.*?-->[ \t]*\n', re.M)


class Doc:
    """One Markdown file: its tokens, its headings and the page it becomes."""

    def __init__(self, md, src_rel: str, out_rel: str, extra: str = ''):
        self.src_rel = src_rel                       # repository-relative, posix
        self.out_rel = out_rel                       # site-relative, posix
        with open(os.path.join(ROOT, src_rel), encoding='utf-8') as f:
            text = f.read()
        # a line that is one HTML comment (the guide's <!-- example: id --> markers) is for the source only: with
        # html off, markdown-it would print it as text
        text = _COMMENT_LINE.sub('', text)
        self.tokens = md.parse(text + extra)
        self.md = md
        self.ids: set[str] = set()
        self.alias: dict[str, str] = {}              # GitHub slug of a heading's text → its id
        self.headings: list[tuple[int, str, str]] = []
        self.title = ''
        self._headings()

    def _headings(self):
        toks = self.tokens
        for i, tok in enumerate(toks):
            if tok.type != 'heading_open':
                continue
            inline = toks[i + 1]
            level = int(tok.tag[1])
            explicit = None
            kids = inline.children or []
            if kids and kids[-1].type == 'text':
                m = re.search(r'\s*\{#([A-Za-z0-9_.:-]+)\}\s*$', kids[-1].content)
                if m:
                    explicit = m.group(1)
                    kids[-1].content = kids[-1].content[:m.start()]
                    inline.content = re.sub(r'\s*\{#[A-Za-z0-9_.:-]+\}\s*$', '', inline.content)
            text = _inline_text(inline).strip()
            hid = explicit or gh_slug(text) or f'h{i}'
            base, n = hid, 1
            while hid in self.ids:
                hid = f'{base}-{n}'
                n += 1
            self.ids.add(hid)
            self.alias.setdefault(gh_slug(text), hid)
            tok.attrSet('id', hid)
            if level == 1 and not self.title:
                self.title = text
            self.headings.append((level, hid, text))

    def render(self) -> str:
        return self.md.renderer.render(self.tokens, self.md.options, {})


def _site_target(repo_path: str) -> str | None:
    """The site page a repository Markdown file becomes, or None when it has no page."""
    if repo_path == 'docs/overview.md':
        return 'docs.html'
    if repo_path == 'docs/user-guide.md':
        return 'guide.html'
    m = re.fullmatch(r'docs/(models|guide|reference)/([A-Za-z0-9_.-]+)\.md', repo_path)
    if m:
        return f'{m.group(1)}/{m.group(2)}.html'
    return None


def _relative(target: str, from_page: str) -> str:
    rel = posixpath.relpath(target, posixpath.dirname(from_page) or '.')
    return rel


class Linker:
    """Rewrites the links of every Doc once all Docs (and so all heading ids) are known."""

    def __init__(self, docs: list[Doc]):
        self.by_out = {d.out_rel: d for d in docs}
        self.warnings: list[str] = []
        self.images: dict[str, str] = {}             # repository image → its copy's site path (copied by build_pages)

    def image(self, src: str, doc: Doc) -> str | None:
        """An image of the repository is copied into the site (site/docs-media/…; site/media holds the JPEG
        screenshots of --media) and named relative to the page; a
        page may load nothing from another site, and a GitHub blob URL is a web page, not an image."""
        if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', src):
            return src
        repo_path = posixpath.normpath(posixpath.join(posixpath.dirname(doc.src_rel), src.partition('#')[0]))
        if repo_path.startswith('../') or not os.path.isfile(os.path.join(ROOT, repo_path)):
            self.warnings.append(f'{doc.src_rel}: the image "{src}" names a file that is not there (left out)')
            return None
        rel = repo_path[len('docs/media/'):] if repo_path.startswith('docs/media/') else repo_path.replace('/', '-')
        site_rel = 'docs-media/' + rel
        self.images[repo_path] = site_rel
        return _relative(site_rel, doc.out_rel)

    def href(self, href: str, doc: Doc) -> str | None:
        if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', href):      # http:, https:, mailto: …
            return href
        path, _, frag = href.partition('#')
        if not path:
            return '#' + self._fragment(doc, frag, href) if frag else href
        repo_path = posixpath.normpath(posixpath.join(posixpath.dirname(doc.src_rel), path))
        if repo_path.startswith('../') or repo_path == '..':
            self.warnings.append(f'{doc.src_rel}: the link "{href}" leaves the repository (shown as text)')
            return None
        page = _site_target(repo_path)
        if page is None:
            full = os.path.join(ROOT, repo_path)
            if not os.path.exists(full):
                self.warnings.append(f'{doc.src_rel}: the link "{href}" names a file that is not there (shown as text)')
                return None
            url = (TREE_URL if os.path.isdir(full) else BLOB_URL) + repo_path
            return url + ('#' + frag if frag else '')
        if page not in self.by_out:
            self.warnings.append(f'{doc.src_rel}: the link "{href}" names a page the site does not build '
                                 '(shown as text)')
            return None
        rel = _relative(page, doc.out_rel)
        if frag:
            f = self._fragment(self.by_out[page], frag, href, doc)
            return rel + ('#' + f if f else '')
        return rel

    def _fragment(self, target: Doc, frag: str, href: str, source: Doc | None = None) -> str:
        if frag in target.ids:
            return frag
        if frag in target.alias:
            return target.alias[frag]
        self.warnings.append(f'{(source or target).src_rel}: the link "{href}" names a heading that '
                             f'{target.src_rel} does not have (links to the page instead)')
        return ''

    def target(self, href: str, doc: Doc) -> Doc | None:
        """The Doc a relative link names, if it names one."""
        if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', href) or href.startswith('#'):
            return None
        path = href.partition('#')[0]
        page = _site_target(posixpath.normpath(posixpath.join(posixpath.dirname(doc.src_rel), path)))
        return self.by_out.get(page) if page else None

    def apply(self, doc: Doc):
        for tok in doc.tokens:
            if tok.type != 'inline' or not tok.children:
                continue
            kids = tok.children
            out = []
            drop_close = 0
            for i, c in enumerate(kids):
                if c.type == 'link_open':
                    old = c.attrGet('href') or ''
                    new = self.href(old, doc)
                    if new is None:
                        drop_close += 1
                        continue
                    c.attrSet('href', new)
                    # a link whose text is the file name ("models/robots.md") reads as the page's title on the site
                    t = self.target(old, doc)
                    nxt = kids[i + 1] if i + 1 < len(kids) else None
                    if (t is not None and t.title and nxt is not None and nxt.type == 'text'
                            and i + 2 < len(kids) and kids[i + 2].type == 'link_close'
                            and nxt.content.strip().endswith('.md')
                            and posixpath.basename(nxt.content.strip()) == posixpath.basename(old.partition('#')[0])):
                        nxt.content = t.title
                elif c.type == 'link_close' and drop_close:
                    drop_close -= 1
                    continue
                elif c.type == 'image':
                    new = self.image(c.attrGet('src') or '', doc)
                    if new is None:
                        continue
                    c.attrSet('src', new)
                out.append(c)
            tok.children = out


# ------------------------------------------------------------------------------------------------ the page frame

ICON_DOWN = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
             'stroke-linejoin="round" aria-hidden="true"><path d="M12 4v11M7 10l5 5 5-5M5 20h14"/></svg>')


def nav_html(prefix: str, current: str | None) -> str:
    links = []
    for label, target, key in NAV:
        cur = ' aria-current="page"' if key and key == current else ''
        links.append(f'      <a href="{prefix}{target}"{cur}>{label}</a>')
    links.append(f'      <a href="{REPO_URL}">GitHub</a>')
    return (f'<nav class="nav" aria-label="Main">\n  <div class="wrap">\n'
            f'    <a class="brand" href="{prefix}index.html"><img src="{prefix}assets/img/logo.png" alt="" width="26" '
            f'height="26">DexVantage</a>\n'
            f'    <button class="nav-toggle" type="button" aria-expanded="false" '
            f'aria-controls="nav-links">Menu</button>\n'
            f'    <div class="nav-links" id="nav-links">\n' + '\n'.join(links) + '\n    </div>\n  </div>\n</nav>\n')


def footer_html(prefix: str) -> str:
    return f'''<footer>
  <div class="wrap">
    <div>
      <a class="brand" href="{prefix}index.html"><img src="{prefix}assets/img/logo.png" alt="" width="22"
        height="22">DexVantage</a>
      <p class="fine">DexVantage is free software under the GNU General Public License, version 3 or later
        (<a href="{BLOB_URL}LICENSE">GPL-3.0-or-later</a>). The screenshots were taken from the application by
        <code>scripts/screens.py</code>. Robot, camera and policy names describe the bundled samples and the hardware
        library and belong to their owners; no affiliation is implied. This site loads nothing from other sites.</p>
    </div>
    <div class="links">
      <a href="{prefix}guide.html">User guide</a>
      <a href="{prefix}docs.html">Models and methods</a>
      <a href="{prefix}methods.html">Technical reference</a>
      <a href="{prefix}validation.html">Validation report</a>
      <a href="{RELEASES_URL}">Releases</a>
      <a href="{BLOB_URL}CITATION.cff">Cite</a>
      <a href="{REPO_URL}">GitHub</a>
      <a href="{REPO_URL}/issues">Report an issue</a>
    </div>
  </div>
</footer>
'''


def page_html(*, out_rel: str, title: str, description: str, current: str | None, body: str,
              toc: str = '', extra_head: str = '') -> str:
    prefix = '../' * out_rel.count('/')
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(description)}">
<meta name="theme-color" content="#0f0f0e">
<link rel="icon" type="image/png" href="{prefix}assets/img/favicon.png">
<link rel="apple-touch-icon" href="{prefix}assets/img/apple-touch-icon.png">
<link rel="stylesheet" href="{prefix}assets/css/site.css">{extra_head}
</head>
<body>
<a class="skip" href="#doc">Skip to content</a>
{nav_html(prefix, current)}
<div class="docs">
{toc}
  <main class="doc" id="doc">
{body}
  </main>
</div>
{footer_html(prefix)}<script src="{prefix}assets/js/site.js"></script>
</body>
</html>
'''


def toc_html(doc: Doc, levels=(2,), label: str = 'On this page', extra: list[tuple[str, str]] = ()) -> str:
    rows = [f'    <p>{html.escape(label)}</p>']
    for level, hid, text in doc.headings:
        if level in levels:
            cls = ' class="sub"' if level > min(levels) else ''
            rows.append(f'    <a href="#{hid}"{cls}>{html.escape(text)}</a>')
    for text, href in extra:
        rows.append(f'    <a href="{href}">{html.escape(text)}</a>')
    return '  <aside class="toc" aria-label="On this page">\n' + '\n'.join(rows) + '\n  </aside>'


def _first_paragraph(doc: Doc) -> str:
    toks = doc.tokens
    for i, tok in enumerate(toks):
        if tok.type == 'paragraph_open' and i + 1 < len(toks):
            text = re.sub(r'\s+', ' ', _inline_text(toks[i + 1])).strip()
            if text:
                return text
    return ''


def _short(text: str, n: int = 220) -> str:
    if len(text) <= n:
        return text
    cut = text[:n].rsplit(' ', 1)[0]
    return cut.rstrip(',;:') + ' …'


def _ordered(names: list[str], text: str, folder: str) -> list[str]:
    """Names in the order the parent document first links them, then the rest alphabetically."""
    first = {}
    for m in re.finditer(rf'\]\({folder}/([A-Za-z0-9_.-]+)\.md', text):
        first.setdefault(m.group(1), m.start())
    return sorted(names, key=lambda n: (n not in first, first.get(n, 0), n))


# ------------------------------------------------------------------------------------------------ the build

def build_pages(site: str = SITE) -> list[str]:
    """Write docs.html, guide.html, methods.html and the model and worked-example pages. Returns the warnings."""
    md = _markdown()
    overview_text = open(os.path.join(DOCS, 'overview.md'), encoding='utf-8').read()
    guide_text = open(os.path.join(DOCS, 'user-guide.md'), encoding='utf-8').read()
    models = _ordered([f[:-3] for f in os.listdir(os.path.join(DOCS, 'models')) if f.endswith('.md')],
                      overview_text, 'models')
    examples = _ordered([f[:-3] for f in os.listdir(os.path.join(DOCS, 'guide')) if f.endswith('.md')],
                        guide_text, 'guide')

    ref_folder = os.path.join(DOCS, 'reference')
    refs = sorted(f[:-3] for f in os.listdir(ref_folder) if f.endswith('.md')) if os.path.isdir(ref_folder) else []
    model_docs = [Doc(md, f'docs/models/{n}.md', f'models/{n}.html') for n in models]
    ref_docs = [Doc(md, f'docs/reference/{n}.md', f'reference/{n}.html') for n in refs]
    example_docs = [Doc(md, f'docs/guide/{n}.md', f'guide/{n}.html') for n in examples]
    listing = '\n\n## Worked examples {#worked-examples}\n\n' + \
        'One page per feature, each a worked example on a sample project.\n\n' + \
        '\n'.join(f'- [{d.title or n}](guide/{n}.md)' for n, d in zip(examples, example_docs)) + '\n'
    overview = Doc(md, 'docs/overview.md', 'docs.html')
    guide = Doc(md, 'docs/user-guide.md', 'guide.html', extra=listing)
    docs = [overview, guide] + model_docs + example_docs + ref_docs
    linker = Linker(docs)
    for d in docs:
        linker.apply(d)
    for repo_path, site_rel in sorted(linker.images.items()):
        dest = os.path.join(site, *site_rel.split('/'))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(os.path.join(ROOT, repo_path), dest)

    pages = {}
    pages['docs.html'] = page_html(
        out_rel='docs.html', title='Models and methods · DexVantage', current='docs',
        description='Every model DexVantage uses, written out with its equations, references, validation and known '
                    'limits: kinematics, lens models, occlusion, coverage, depth validity, grasp self-occlusion, VLA '
                    'tokens, timing, bandwidth, calibration, ground truth, datasets and the research tools.',
        toc=toc_html(overview), body='    <div class="eyebrow">Documentation</div>\n' + overview.render())
    pages['guide.html'] = page_html(
        out_rel='guide.html', title='User guide · DexVantage', current='guide',
        description='How to use DexVantage step by step: the window, batch mode, robots, sensors, grasps, hardware, '
                    'calibration, ground truth, teleop, lighting, timing, data, requirements, export, the assistant, '
                    'the samples and the research tools, each with a worked example.',
        toc=toc_html(guide, levels=(2,)), body='    <div class="eyebrow">Guide</div>\n' + guide.render())
    for d in model_docs + example_docs + ref_docs:
        is_model = not d.out_rel.startswith('guide/')
        back = ('../methods.html', 'All model documents') if is_model else ('../guide.html#worked-examples',
                                                                          'All worked examples')
        if d.out_rel.startswith('reference/'):
            back = ('../methods.html#reference', 'All reference pages')
        crumbs = (f'    <p class="crumbs"><a href="{back[0]}">{back[1]}</a> · source: '
                  f'<a href="{BLOB_URL}{d.src_rel}"><code>{d.src_rel}</code></a></p>\n')
        eyebrow = 'Technical reference' if is_model else 'Worked example'
        pages[d.out_rel] = page_html(
            out_rel=d.out_rel, title=f'{d.title or d.out_rel} · DexVantage', current=None,
            description=_short(_first_paragraph(d), 300) or d.title,
            toc=toc_html(d, levels=(2,), extra=[(back[1], back[0])]),
            body=f'    <div class="eyebrow">{eyebrow}</div>\n' + crumbs + d.render())

    rows = []
    for d in model_docs:
        rows.append(f'      <li><a href="{d.out_rel}">{html.escape(d.title or d.out_rel)}</a>'
                    f'<span>{html.escape(_short(_first_paragraph(d)))}</span></li>')
    body = ('    <div class="eyebrow">Technical reference</div>\n'
            '    <h1 id="models">The models behind DexVantage</h1>\n'
            '    <p class="intro">One document per model: what it computes and from which inputs, its equations as the '
            'code evaluates them, the references it follows, the validation checks that compare it with an independent '
            'reference and the agreement they measured, and its known limits. '
            '<a href="docs.html">Models and methods</a> gives the overview of all of them; the '
            '<a href="validation.html">validation report</a> gives every check with its error and tolerance.</p>\n'
            '    <ul class="index">\n' + '\n'.join(rows) + '\n    </ul>\n')
    if ref_docs:
        body += ('    <h2 id="reference">Reference</h2>\n    <ul class="index">\n' + '\n'.join(
            f'      <li><a href="{d.out_rel}">{html.escape(d.title or d.out_rel)}</a>'
            f'<span>{html.escape(_short(_first_paragraph(d)))}</span></li>' for d in ref_docs) + '\n    </ul>\n')
    pages['methods.html'] = page_html(
        out_rel='methods.html', title='Technical reference · DexVantage', current='methods',
        description='The model documents behind DexVantage: equations, references, validation and known limits of '
                    'every study, reader and exporter.',
        toc='  <aside class="toc" aria-label="On this page">\n    <p>Reference</p>\n'
            '    <a href="#models">Model documents</a>\n'
            + ('    <a href="#reference">Reference</a>\n' if ref_docs else '') +
            '    <a href="docs.html">Models and methods</a>\n'
            '    <a href="validation.html">Validation report</a>\n  </aside>',
        body=body)

    for sub in ('models', 'guide', 'reference'):          # pages of removed Markdown files go too
        folder = os.path.join(site, sub)
        if os.path.isdir(folder):
            for f in os.listdir(folder):
                if f.endswith('.html') and f'{sub}/{f}' not in pages:
                    os.remove(os.path.join(folder, f))
    for rel, text in pages.items():
        path = os.path.join(site, *rel.split('/'))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(text)
    return linker.warnings


# ------------------------------------------------------------------------------------------------ validation page

_SUP = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')


def fmt_number(x) -> str:
    """A figure as the app writes it: 0, four significant digits, or m × 10ⁿ outside 0.001–100 000."""
    if x is None:
        return '—'
    try:
        x = float(x)
    except (TypeError, ValueError):
        return html.escape(str(x))
    if x != x:
        return '—'
    if x == 0:
        return '0'
    if x in (float('inf'), float('-inf')):
        return '∞' if x > 0 else '−∞'
    sign = '−' if x < 0 else ''
    a = abs(x)
    if 1e-3 <= a < 1e3:
        return sign + f'{a:.4g}'
    if 1e3 <= a < 1e5:
        return sign + f'{round(a):,}'.replace(',', ' ')
    m, e = f'{a:.2e}'.split('e')
    m = m.rstrip('0').rstrip('.')
    return f'{sign}{m} × 10{str(int(e)).translate(_SUP)}'


def _overview_ids() -> set[str]:
    path = os.path.join(SITE, 'docs.html')
    if not os.path.exists(path):
        return set()
    return collect(path).ids


def build_validation(report_dir: str, site: str = SITE) -> dict:
    """Write validation.html from a --validate report folder; copy its CSV, JSON and figures. Returns the summary."""
    src = os.path.join(report_dir, 'validation.json')
    if not os.path.exists(src):
        raise SiteError(f'There is no validation report in "{report_dir}": run '
                        f'"dexvantage --validate {report_dir}" first.')
    try:
        with open(src, encoding='utf-8') as f:
            rep = json.load(f)
    except (OSError, ValueError) as e:
        raise SiteError(f'The validation report "{src}" could not be read ({e.__class__.__name__}).') from None
    if rep.get('schema') != 'dexvantage.validation/1' or not isinstance(rep.get('models'), list):
        raise SiteError(f'"{src}" is not a DexVantage validation report (schema dexvantage.validation/1).')

    out_dir = os.path.join(site, 'validation')
    fig_dir = os.path.join(out_dir, 'figures')
    if os.path.isdir(fig_dir):
        shutil.rmtree(fig_dir)
    os.makedirs(fig_dir, exist_ok=True)
    for name in ('validation.json', 'validation.csv'):
        p = os.path.join(report_dir, name)
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(out_dir, name))

    anchors = _overview_ids()
    summ = rep.get('summary', {})
    env = rep.get('environment', {})
    refs = rep.get('references', {}) or {}
    used_refs: list[str] = []
    toc = ['  <aside class="toc" aria-label="On this page">', '    <p>Report</p>',
           '    <a href="#summary">Summary</a>', '    <p>Models</p>']
    sections = []
    for m in rep['models']:
        mid = 'm-' + re.sub(r'[^A-Za-z0-9_-]', '-', str(m.get('id', '')))
        checks = m.get('checks', [])
        passed = sum(1 for c in checks if c.get('passed'))
        toc.append(f'    <a href="#{mid}">{html.escape(m.get("title", m.get("id", "")))}</a>')
        anchor = m.get('methods_anchor')
        model_link = f'docs.html#{anchor}' if anchor in anchors else 'docs.html'
        rows = []
        for c in checks:
            cid = str(c.get('id', ''))
            ok = c.get('passed')
            status = '<span class="ok">pass</span>' if ok else '<span class="no">fail</span>'
            fig = ''
            if c.get('figure'):
                svg = os.path.join(report_dir, 'figures', cid + '.svg')
                if os.path.exists(svg):
                    shutil.copyfile(svg, os.path.join(fig_dir, cid + '.svg'))
                    fig = f' <a href="validation/figures/{html.escape(cid)}.svg">figure</a>'
            rlinks = []
            for r in c.get('references') or []:
                if r in refs:
                    if r not in used_refs:
                        used_refs.append(r)
                    rlinks.append(f'<a href="#ref-{html.escape(r)}">[{used_refs.index(r) + 1}]</a>')
            note = f'<br><span class="muted">{html.escape(c["note"])}</span>' if c.get('note') else ''
            unit = c.get('unit') or ''
            unit = '' if unit == '(unitless)' else f'<span class="unit">{html.escape(unit)}</span>'
            rows.append(
                f'      <tr id="c-{html.escape(cid)}"><td><code>{html.escape(cid)}</code><br>'
                f'{html.escape(c.get("title", ""))}</td>'
                f'<td><b>Compared:</b> {html.escape(c.get("quantity", ""))}<br><b>Reference:</b> '
                f'{html.escape(c.get("reference", ""))} {" ".join(rlinks)}{fig}{note}</td>'
                f'<td class="num">{fmt_number(c.get("error"))}{unit}</td>'
                f'<td class="num">{fmt_number(c.get("tolerance"))}{unit}</td><td>{status}</td></tr>')
        sections.append(
            f'    <h2 id="{mid}">{html.escape(m.get("title", ""))}</h2>\n'
            f'    <p>{passed} of {len(checks)} checks pass. '
            f'Code: <code>{html.escape(str(m.get("modules", "")))}</code>. '
            f'The model: <a href="{model_link}">{html.escape(m.get("title", ""))} in models and methods</a>.</p>\n'
            '    <table class="val">\n      <thead><tr><th>Check</th><th>What is compared, and with what</th>'
            '<th>Error</th><th>Tolerance</th><th>Result</th></tr></thead>\n      <tbody>\n'
            + '\n'.join(rows) + '\n      </tbody>\n    </table>\n')
    toc.append('    <p>Reference</p>')
    toc.append('    <a href="#references">References</a>')
    toc.append('  </aside>')

    ref_items = '\n'.join(f'      <li id="ref-{html.escape(r)}">{html.escape(refs[r])}</li>' for r in used_refs)
    n, ok = summ.get('checks', 0), summ.get('passed', 0)
    failed = summ.get('failed') or []
    env_txt = ', '.join(f'{k} {v}' for k, v in env.items() if k != 'platform')
    plat = env.get('platform', '')
    date = str(rep.get('created_utc', ''))[:10]
    secs = rep.get('seconds')
    settings = rep.get('settings') or {}
    fail_txt = ('' if not failed else
                f'<p class="callout warn">Failed: {html.escape(", ".join(map(str, failed)))}.</p>\n')
    body = f'''    <div class="eyebrow">Validation</div>
    <h1>The validation report</h1>
    <p class="intro">Every model in DexVantage is checked against an independent reference: a closed form written out
      again, another library, a published table, brute-force ray casting, Monte Carlo or MuJoCo renders. Each row gives
      what was compared, against what, the error measured and the tolerance with its reason. The page is generated
      from the report that <code>dexvantage --validate DIR</code> writes
      (<code>scripts/build_site.py --validation</code>).</p>
    <h2 id="summary">Summary</h2>
    <div class="facts small">
      <div class="fact"><b>{ok} of {n}</b><span>checks pass</span></div>
      <div class="fact"><b>{len(rep['models'])}</b><span>models checked</span></div>
      <div class="fact"><b>{fmt_number(secs)} s</b><span>for the full run</span></div>
      <div class="fact"><b>{html.escape(date)}</b><span>date of the run (UTC)</span></div>
    </div>
{fail_txt}    <p>Run on {html.escape(plat)} with {html.escape(env_txt)}. Settings: seed
      {html.escape(str(settings.get("seed", "—")))}, {"quick" if settings.get("quick") else "full"} run.
      Frame: {html.escape(str(rep.get("frame", "")))}.</p>
    <p class="note">{html.escape(str(rep.get("limits", "")))}</p>
    <p>Download the report: <a href="validation/validation.csv">CSV</a> ·
      <a href="validation/validation.json">JSON</a> (with its provenance block). Run it yourself:
      <code>dexvantage --validate out/validation</code>.</p>
{"".join(sections)}    <h2 id="references">References</h2>
    <ol class="refs">
{ref_items}
    </ol>
'''
    text = page_html(out_rel='validation.html', title='Validation report · DexVantage', current='validation',
                     description=f'{ok} of {n} validation checks of DexVantage pass against independent references: '
                                 'the error measured and the tolerance of every check.',
                     toc='\n'.join(toc), body=body)
    with open(os.path.join(site, 'validation.html'), 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)
    return {'checks': n, 'passed': ok, 'models': len(rep['models']), 'date': date}


# ------------------------------------------------------------------------------------------------ media

def build_media(screens: str, site: str = SITE, docs_media: str = os.path.join(DOCS, 'media')) -> list[str]:
    """Convert the listed screenshots to JPEG under site/media (all) and docs/media (the README's), each < 400 KB."""
    try:
        from PIL import Image
    except ImportError:
        raise SiteError('The media step needs Pillow (python -m pip install pillow).') from None
    os.makedirs(os.path.join(site, 'media'), exist_ok=True)
    os.makedirs(docs_media, exist_ok=True)
    written = []
    for src, name, readme in MEDIA:
        p = os.path.join(screens, src)
        if not os.path.exists(p):
            raise SiteError(f'The screenshot "{src}" is not in "{screens}": run "python scripts/screens.py" first.')
        im = Image.open(p).convert('RGB')
        dests = [os.path.join(site, 'media', name)] + ([os.path.join(docs_media, name)] if readme else [])
        for q in (86, 80, 72, 64, 56):
            tmp = dests[0] + '.part'
            im.save(tmp, 'JPEG', quality=q, optimize=True, progressive=True)
            if os.path.getsize(tmp) < MEDIA_LIMIT:
                break
        else:
            os.remove(tmp)
            raise SiteError(f'"{src}" stays over 400 KB as a JPEG.')
        for d in dests[1:]:
            shutil.copyfile(tmp, d)
        os.replace(tmp, dests[0])
        written += dests
    return written


def build_assets(site: str = SITE):
    """The fonts (IBM Plex, OFL) and the icon, from the app's own resources."""
    fonts = os.path.join(ROOT, 'dexvantage', 'resources', 'fonts')
    dst = os.path.join(site, 'assets', 'fonts')
    os.makedirs(dst, exist_ok=True)
    for f in ('IBMPlexSans-Regular.ttf', 'IBMPlexSans-Medium.ttf', 'IBMPlexSans-SemiBold.ttf',
              'IBMPlexMono-Regular.ttf', 'IBMPlexMono-Medium.ttf', 'OFL.txt'):
        shutil.copyfile(os.path.join(fonts, f), os.path.join(dst, f))
    img = os.path.join(site, 'assets', 'img')
    os.makedirs(img, exist_ok=True)
    icon = os.path.join(ROOT, 'dexvantage', 'resources', 'icons', 'dexvantage.png')
    try:
        from PIL import Image
    except ImportError:
        for f in ('logo.png', 'favicon.png', 'apple-touch-icon.png'):
            shutil.copyfile(icon, os.path.join(img, f))
        return
    im = Image.open(icon).convert('RGBA')
    for f, n in (('logo.png', 64), ('favicon.png', 48), ('apple-touch-icon.png', 180)):
        im.resize((n, n), Image.LANCZOS).save(os.path.join(img, f), optimize=True)


# ------------------------------------------------------------------------------------------------ checks

VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'source', 'track', 'wbr'}


class _Collector(HTMLParser):
    """Ids, links and nesting errors of one page (html.parser, standard library)."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.dup_ids: list[str] = []
        self.links: list[tuple[str, str]] = []     # (attribute, value)
        self.errors: list[str] = []
        self.stack: list[tuple[str, int]] = []
        self.saw_doctype = False

    def handle_decl(self, decl):
        if decl.lower().startswith('doctype'):
            self.saw_doctype = True

    def _attrs(self, tag, attrs):
        for k, v in attrs:
            if k == 'id' and v is not None:
                if v in self.ids:
                    self.dup_ids.append(v)
                self.ids.add(v)
            if k == 'name' and tag == 'a' and v:
                self.ids.add(v)
            if (k == 'href' and tag in ('a', 'link')) or (k in ('src', 'poster') and tag in ('img', 'script', 'source',
                                                                                              'video')):
                if v is not None:
                    self.links.append((k, v))

    def handle_starttag(self, tag, attrs):
        self._attrs(tag, attrs)
        if tag not in VOID:
            self.stack.append((tag, self.getpos()[0]))

    def handle_startendtag(self, tag, attrs):
        self._attrs(tag, attrs)

    def handle_endtag(self, tag):
        if tag in VOID:
            self.errors.append(f'line {self.getpos()[0]}: end tag for the void element <{tag}>')
            return
        if not self.stack:
            self.errors.append(f'line {self.getpos()[0]}: </{tag}> with nothing open')
            return
        top, line = self.stack[-1]
        if top == tag:
            self.stack.pop()
            return
        self.errors.append(f'line {self.getpos()[0]}: </{tag}> closes <{top}> opened on line {line}')
        if any(t == tag for t, _ in self.stack):
            while self.stack and self.stack[-1][0] != tag:
                self.stack.pop()
            self.stack.pop()


def collect(path: str) -> _Collector:
    c = _Collector()
    with open(path, encoding='utf-8') as f:
        c.feed(f.read())
    c.close()
    return c


def site_pages(site: str = SITE) -> list[str]:
    out = []
    for root, dirs, files in os.walk(site):
        dirs.sort()
        out += [os.path.join(root, f) for f in sorted(files) if f.endswith('.html') and
                not os.path.relpath(os.path.join(root, f), site).replace(os.sep, '/').startswith('validation/')]
    return out


def check_wellformed(site: str = SITE) -> list[str]:
    problems = []
    for p in site_pages(site):
        rel = os.path.relpath(p, site).replace(os.sep, '/')
        c = collect(p)
        if not c.saw_doctype:
            problems.append(f'{rel}: no <!doctype html>')
        problems += [f'{rel}: {e}' for e in c.errors]
        problems += [f'{rel}: <{t}> opened on line {line} is never closed' for t, line in c.stack]
        problems += [f'{rel}: the id "{i}" is used twice' for i in c.dup_ids]
    return problems


def check_links(site: str = SITE) -> list[str]:
    """Every internal link and image of every page must name a file under site/ and, with a #fragment in an HTML
    page, an id in it. External links (with a scheme) are not fetched."""
    problems = []
    cache: dict[str, _Collector] = {}

    def ids(path):
        if path not in cache:
            cache[path] = collect(path)
        return cache[path].ids

    root = os.path.realpath(site)
    for p in site_pages(site):
        rel = os.path.relpath(p, site).replace(os.sep, '/')
        for attr, href in collect(p).links:
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', href) or href.startswith('//'):
                if href.startswith(('http://', 'https://', 'mailto:')):
                    continue
                problems.append(f'{rel}: the link "{href}" uses a scheme the site does not allow')
                continue
            path, _, frag = href.partition('#')
            path = path.split('?', 1)[0]
            if not path:
                target = p
            else:
                target = os.path.normpath(os.path.join(os.path.dirname(p), *path.split('/')))
                if path.endswith('/') or os.path.isdir(target):
                    target = os.path.join(target, 'index.html')
            if not os.path.realpath(target).startswith(root + os.sep):
                problems.append(f'{rel}: the link "{href}" leaves the site')
                continue
            if not os.path.isfile(target):
                problems.append(f'{rel}: the link "{href}" names a file that is not there')
                continue
            if frag and target.endswith('.html') and frag not in ids(target):
                problems.append(f'{rel}: the link "{href}" names an id the page does not have')
    return problems


def check_media(site: str = SITE, docs_media: str = os.path.join(DOCS, 'media')) -> list[str]:
    problems = []
    for folder in (os.path.join(site, 'media'), os.path.join(site, 'docs-media'), docs_media):
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            if os.path.getsize(os.path.join(folder, f)) >= MEDIA_LIMIT:
                problems.append(f'{os.path.relpath(os.path.join(folder, f), ROOT)} is 400 KB or more')
    return problems


def check_site(site: str = SITE) -> list[str]:
    return check_wellformed(site) + check_links(site) + check_media(site)


# ------------------------------------------------------------------------------------------------ command line

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Build and check the DexVantage website (site/).')
    ap.add_argument('--validation', nargs='?', const=os.path.join(ROOT, 'out', 'validation'), metavar='DIR',
                    help='also write validation.html from a --validate report folder (default out/validation)')
    ap.add_argument('--media', nargs='?', const=os.path.join(ROOT, 'out', 'screens'), metavar='DIR',
                    help='also convert the screenshots from DIR (default out/screens) into site/media and docs/media')
    ap.add_argument('--check', action='store_true', help='only check the site: links, HTML and image sizes')
    ap.add_argument('--site', default=SITE, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    try:
        if not a.check:
            build_assets(a.site)
            if a.media:
                print(f'media: {len(build_media(a.media, a.site))} files')
            for w in build_pages(a.site):
                print('note:', w)
            if a.validation:
                s = build_validation(a.validation, a.site)
                print(f'validation.html: {s["passed"]} of {s["checks"]} checks, {s["models"]} models, {s["date"]}')
        problems = check_site(a.site)
    except SiteError as e:
        print(e, file=sys.stderr)
        return 2
    for p in problems:
        print('problem:', p)
    verdict = 'no problems' if not problems else f'{len(problems)} problems'
    print(f'{len(site_pages(a.site))} pages checked: {verdict}')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
