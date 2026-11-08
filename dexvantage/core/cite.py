"""Citations: a BibTeX file of every source a project depends on, and of DexVantage itself.

    entries(project, references='checks') -> [Entry]      Entry(type, key, fields, group)
    bibtex(project, references='checks') -> str           the .bib text
    write(project, path, references='checks') -> {'path', 'entries', 'keys'}
    self_entry() -> Entry                                 DexVantage, from CITATION.cff

What is cited, in this order (one group comment each):

1. DexVantage itself, from CITATION.cff at the repository root (title, authors, version, licence, repository URL);
   an installed copy without that file falls back to the package's own metadata (name, version, URL).
2. The library entries the project names (sensors of every rig variant, policies, the rig's hosts, cables, lights):
   each source of an entry (a maker's datasheet, a model card, a paper, a code repository) as @misc with its title,
   url and urldate (the date the source was read, from the entry). A source several entries cite (one datasheet
   for a family) is one item; its note names every entry it backs. Its key names the document, not one device
   (shared_key: the first device's maker, a colon and the document's file or page name, e.g.
   intel:realsense-d400-series-datasheet-mar-2026 for the D400 Series datasheet that backs the D405 and the D435),
   so a paper cites it once and no device is cited under another's name. A source one entry cites keeps that
   entry's id as its key (-2, -3, ... for its second and later sources).
3. Robots: a MuJoCo Menagerie model as @misc with the tree URL at the commit the files came from and its licence,
   plus Menagerie's own citation as its README gives it (read 2026-11-07); other robots as @misc with their name,
   licence and import note.
4. Hugging Face Hub datasets the project audits, with their revision.
5. The references the validation checks cite (validation/references.py: papers, standards, software), when
   references='checks' (the default), all of them with 'all', none with 'none'. A reference written as
   'Authors, "Title", venue, year' becomes author, title, howpublished and year; any other is a @misc whose title
   is the reference's text, so nothing is guessed about its parts.

Keys are unique (a repeated key gets -2, -3, ...), made of letters, digits, '-', '_', ':' and '.'. Field values are
UTF-8 (biber reads them as they are; classic BibTeX wants \\usepackage[utf8]{inputenc}); TeX's special characters are
escaped outside url fields. urldate is a biblatex field; classic BibTeX styles ignore it, so the date is also in note.

Known limits: the reference parser splits author lists only at "and" and commas and keeps corporate authors whole in
braces; a reference that names its year only in its title gets that year; local dataset folders are not cited (a
path is not a source others can find); policy figures the user measured (overrides) are not citations.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

MENAGERIE_WEB = 'https://github.com/google-deepmind/mujoco_menagerie'
# MuJoCo Menagerie's own citation, as its README gives it (README.md of google-deepmind/mujoco_menagerie, main,
# read 2026-11-07)
MENAGERIE_CITATION = {'author': 'Zakka, Kevin and Tassa, Yuval and {MuJoCo Menagerie Contributors}',
                      'title': '{MuJoCo Menagerie: A collection of high-quality simulation models for MuJoCo}',
                      'url': 'http://github.com/google-deepmind/mujoco_menagerie', 'year': '2022'}
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_FIELDS = ('url', 'urldate', 'doi')                   # values written without TeX escaping
_KEY_BAD = re.compile(r'[^A-Za-z0-9_:.-]+')


@dataclass
class Entry:
    type: str
    key: str
    fields: dict = field(default_factory=dict)
    group: str = ''


def escape(text: str) -> str:
    """TeX's special characters escaped for a BibTeX field value (braces balanced by construction)."""
    rep = {'\\': r'\textbackslash{}', '{': r'\{', '}': r'\}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
           '_': r'\_', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(rep.get(c, c) for c in str(text))


def _key(s: str) -> str:
    k = _KEY_BAD.sub('-', str(s)).strip('-.:') or 'ref'
    return k if k[0].isalpha() else f'r{k}'


def shared_key(maker_id: str, ref: str) -> str:
    """The key of a source several library entries cite: the first entry's maker (its id's first word), a colon and
    the document's name (the URL's last path part without its extension, else the document's first words), in
    lower case, cut to 48 characters at a word."""
    from urllib.parse import unquote, urlsplit
    ref = str(ref or '').strip()
    if re.match(r'^https?://', ref):
        parts = [x for x in urlsplit(ref).path.split('/') if x]
        name = unquote(parts[-1]) if parts else urlsplit(ref).netloc
        name = re.sub(r'\.(pdf|md|html?|php|aspx?)$', '', name, flags=re.I)
    else:
        name = ' '.join(ref.split()[:6])
    slug = re.sub(r'[^a-z0-9.]+', '-', name.lower()).strip('-.')
    if len(slug) > 48:
        slug = slug[:48].rsplit('-', 1)[0] if '-' in slug[:48] else slug[:48]
    maker = str(maker_id or '').split('-')[0] or 'source'
    return _key(f'{maker}:{slug or "source"}')


def _url_clean(u: str) -> str:
    return u.rstrip('.,;)')


# ====================================================================== DexVantage itself
def _cff() -> dict | None:
    path = os.path.join(REPO_ROOT, 'CITATION.cff')
    if not os.path.isfile(path):
        return None
    try:
        import yaml
        with open(path, encoding='utf-8') as fh:
            d = yaml.safe_load(fh)
        return d if isinstance(d, dict) else None
    except Exception:                                   # noqa: BLE001 — a damaged file falls back to the metadata
        return None


def _cff_author(a: dict) -> str:
    if a.get('family-names'):
        given = a.get('given-names') or ''
        return f'{a["family-names"]}, {given}'.strip(', ')
    return '{' + str(a.get('name') or '') + '}' if ' and ' in str(a.get('name') or '') else str(a.get('name') or '')


def self_entry() -> Entry:
    from .. import __version__
    d = _cff()
    f: dict = {}
    if d:
        f['title'] = '{' + escape(d.get('title') or 'DexVantage') + '}'
        authors = [_cff_author(a) for a in d.get('authors') or [] if isinstance(a, dict)]
        if authors:
            f['author'] = ' and '.join(escape(a) for a in authors)
        f['version'] = escape(str(d.get('version') or __version__))
        if d.get('repository-code') or d.get('url'):
            f['url'] = str(d.get('repository-code') or d.get('url'))
        if d.get('license'):
            f['license'] = escape(str(d['license']))
        if d.get('date-released'):
            f['year'] = str(d['date-released'])[:4]
        f['note'] = escape(f'Version {f["version"]}; cite as CITATION.cff asks')
    else:
        f['title'] = '{DexVantage}'
        f['version'] = escape(__version__)
        try:
            from importlib import metadata
            md = metadata.metadata('dexvantage')
            urls = [u.split(',', 1)[1].strip() for u in md.get_all('Project-URL') or [] if ',' in u]
            if urls:
                f['url'] = urls[0]
        except Exception:                               # noqa: BLE001 — not installed as a package: no URL
            pass
        f['note'] = escape(f'Version {__version__} (CITATION.cff not found beside this copy)')
    return Entry('software', 'dexvantage', f, 'DexVantage')


# ====================================================================== the project's sources
def _library_ids(project) -> list:
    from . import jobs
    ids = [s.model_id for v in project.variants for s in v.sensors if s.model_id]
    ids += jobs.library_ids(project)
    ids += [str(x.get('library_id')) for x in project.workspace.lights if isinstance(x, dict) and x.get('library_id')]
    out = []
    for i in ids:
        if i not in out:
            out.append(i)
    return out


def _library_entries(project) -> tuple[list, list]:
    from . import library as L
    by_url: dict = {}
    order = []
    missing = []
    for mid in _library_ids(project):
        e = L.get(mid)
        if e is None:
            missing.append(mid)
            continue
        for k, s in enumerate(e.sources):
            if not isinstance(s, dict):
                continue
            ref = str(s.get('url') or s.get('document') or '').strip()
            if not ref:
                continue
            if ref not in by_url:
                by_url[ref] = {'entry': e, 'source': s, 'ids': [], 'k': k}
                order.append(ref)
            if e.id not in by_url[ref]['ids']:
                by_url[ref]['ids'].append(e.id)
    out = []
    for ref in order:
        r = by_url[ref]
        e, s = r['entry'], r['source']
        f = {'title': '{' + escape(s.get('title') or e.title) + '}'}
        if e.kind != 'policy':
            f['author'] = '{' + escape(e.maker) + '}'
        if s.get('url'):
            f['url'] = str(s['url'])
        else:
            f['howpublished'] = escape(str(s.get('document')))
        read = str(s.get('read') or '')
        if re.match(r'^\d{4}-\d{2}-\d{2}$', read):
            f['urldate'] = read
        note = f'A source of DexVantage library entr{"y" if len(r["ids"]) == 1 else "ies"} {", ".join(r["ids"])}'
        if s.get('covers') and len(r['ids']) == 1:        # what it covers is said per entry: shared, it is left out
            note += f' (for {" ".join(str(s["covers"]).split())})'
        if read:
            note += f'; read {read}'
        f['note'] = escape(note)
        if len(r['ids']) > 1:                             # a shared source: keyed by the document, not one device
            key = shared_key(r['ids'][0], ref)
        else:
            key = e.id if r['k'] == 0 else f'{e.id}-{r["k"] + 1}'
        out.append(Entry('misc', _key(key), f, 'Library sources (datasheets, model cards, papers)'))
    return out, missing


_COMMIT = re.compile(r'\bcommit ([0-9a-f]{7,40})\b')
_TREE = re.compile(r'mujoco_menagerie/tree/([0-9a-f]{7,40})/([A-Za-z0-9_.-]+)')
_FOLDERS = re.compile(r'folder\(s\) ([A-Za-z0-9_., -]+?);')
_LIC_FOLDER = re.compile(r'Menagerie ([A-Za-z0-9_.-]+)/LICENSE')


def menagerie_models(robot) -> list:
    """[(folder, commit)] of the MuJoCo Menagerie models a robot record names (its source note and licence)."""
    text = f'{robot.source.note} {robot.source.license}'
    if 'menagerie' not in text.lower():
        return []
    out = []
    for c, folder in _TREE.findall(text):
        out.append((folder, c))
    commit = (_COMMIT.findall(text) or [''])[0]
    folders = []
    for m in _FOLDERS.findall(text):
        folders += [x.strip() for x in m.split(',') if x.strip()]
    folders += _LIC_FOLDER.findall(text)
    for f in folders:
        if not any(f == x for x, _ in out):
            out.append((f, commit))
    return out


def _robot_entries(project) -> list:
    out = []
    seen = set()
    any_menagerie = False
    for r in project.robots:
        models = menagerie_models(r)
        for folder, commit in models:
            any_menagerie = True
            if (folder, commit) in seen:
                continue
            seen.add((folder, commit))
            url = f'{MENAGERIE_WEB}/tree/{commit}/{folder}' if commit else f'{MENAGERIE_WEB}/tree/main/{folder}'
            lic = ''
            m = re.search(r'([A-Za-z0-9.+-]+) \((?:MuJoCo )?Menagerie ' + re.escape(folder) + r'/LICENSE',
                          r.source.license)
            if m:
                lic = m.group(1)
            note = f'MuJoCo Menagerie model {folder}' + (f' at commit {commit}' if commit else ' (commit not recorded)')
            if lic:
                note += f'; licence {lic} (the model folder\'s own LICENSE)'
            note += f'; used by robot {r.name}'
            f = {'title': '{' + escape(f'MuJoCo Menagerie: {folder}') + '}',
                 'author': '{MuJoCo Menagerie Contributors}', 'url': url, 'note': escape(note)}
            out.append(Entry('misc', _key(f'menagerie-{folder}'), f, 'Robot models'))
        if not models and (r.source.note or r.source.license or r.source.file):
            note = ' '.join(x for x in (r.source.note, f'Licence: {r.source.license}.' if r.source.license else '')
                            if x)
            f = {'title': '{' + escape(r.name) + '}',
                 'note': escape(' '.join(note.split()) or f'Robot model file {os.path.basename(r.source.file)}')}
            urls = re.findall(r'https?://\S+', note)
            if urls:
                f['url'] = _url_clean(urls[0])
            out.append(Entry('misc', _key(f'robot-{r.name}'), f, 'Robot models'))
    if any_menagerie:
        f = dict(MENAGERIE_CITATION)
        out.insert(0, Entry('software', 'menagerie2022github', f, 'Robot models'))
    return out


def _dataset_entries(project) -> list:
    from .datasets.hub import parse_hub_ref
    out = []
    for d in project.data.get('datasets') or []:
        ref = parse_hub_ref(d.get('path'))
        if ref is None:
            continue
        repo, rev = ref
        f = {'title': '{' + escape(repo) + '}', 'url': f'https://huggingface.co/datasets/{repo}',
             'note': escape(f'Hugging Face Hub dataset, revision {rev}' + (f'; audited as {d["name"]}'
                                                                           if d.get('name') else ''))}
        out.append(Entry('misc', _key(f'hf-{repo.replace("/", "-")}'), f, 'Datasets'))
    return out


# ====================================================================== validation references
_PAPER = re.compile(r'^(?P<authors>[^"]+?), "(?P<title>[^"]+)",?\s*(?P<rest>.*)$', re.S)
_YEAR = re.compile(r'\b(1[89]\d{2}|20\d{2})\b')
_READ = re.compile(r'\(?\bread (\d{4}-\d{2}-\d{2})\b[^)]*\)?')


def _authors(s: str) -> str:
    s = s.strip()
    others = bool(re.search(r'\bet al\.?$', s))
    s = re.sub(r',?\s*et al\.?$', '', s)
    if not re.search(r'\b[A-Z]\.', s):                  # no initials: a corporate author, kept whole
        return '{' + escape(s) + '}'
    parts = [p.strip() for p in re.split(r',\s*(?:and\s+)?|\s+and\s+', s) if p.strip()]
    return ' and '.join(escape(p) for p in parts) + (' and others' if others else '')


def reference_entry(key: str, text: str) -> Entry:
    """A validation/references.py entry as BibTeX (see the module docstring for how it is read)."""
    t = ' '.join(str(text).split())
    urls = [_url_clean(u) for u in re.findall(r'https?://[^\s,]+', t)]
    read = _READ.search(t)
    f: dict = {}
    m = _PAPER.match(t)
    if m:
        rest = m.group('rest')
        f['author'] = _authors(m.group('authors'))
        f['title'] = '{' + escape(m.group('title').strip().rstrip(',')) + '}'
        venue = rest
        for u in urls:
            venue = venue.replace(u, '')
        venue = _READ.sub('', venue)
        venue = re.sub(r'\s+', ' ', venue).strip(' ,.;')
        years = _YEAR.findall(venue)
        if years:
            f['year'] = years[-1]
        if venue:
            f['howpublished'] = escape(venue)
    else:
        body = t
        for u in urls:
            body = body.replace(u, '')
        body = _READ.sub('', body)
        body = re.sub(r'\s+', ' ', re.sub(r',\s*(,|\.|$)', r'\1', body)).strip(' ,.;')
        f['title'] = '{' + escape(body) + '}'
    if urls:
        f['url'] = urls[0]
    if read:
        f['urldate'] = read.group(1)
    f['note'] = escape(f'Cited by DexVantage\'s validation checks (reference {key})'
                       + (f'; read {read.group(1)}' if read else ''))
    return Entry('misc', _key(key), f, 'References of the validation checks (validation/references.py)')


def _reference_entries(which: str) -> list:
    from ..validation.references import REFERENCES
    if which == 'none':
        return []
    if which == 'all':
        keys = list(REFERENCES)
    elif which == 'checks':
        from ..validation import suite as VS
        VS.load_checks()
        cited = {r for s in VS.CHECKS for r in s.refs}
        keys = [k for k in REFERENCES if k in cited]
    else:
        raise ValueError(f'references must be "checks", "all" or "none", not {which!r}')
    return [reference_entry(k, REFERENCES[k]) for k in keys]


# ====================================================================== the file
def entries(project, references: str = 'checks') -> list:
    """Every entry, keys made unique (DexVantage first)."""
    out = [self_entry()]
    if project is not None:
        lib, _missing = _library_entries(project)
        out += lib + _robot_entries(project) + _dataset_entries(project)
    out += _reference_entries(references)
    seen: dict = {}
    for e in out:
        base = e.key
        n = seen.get(base.lower(), 0)
        seen[base.lower()] = n + 1
        if n:
            k = f'{base}-{n + 1}'
            while k.lower() in seen:
                n += 1
                k = f'{base}-{n + 1}'
            seen[k.lower()] = 1
            e.key = k
    return out


def format_entry(e: Entry) -> str:
    w = max(len(k) for k in e.fields) if e.fields else 0
    lines = [f'@{e.type}{{{e.key},']
    for k, v in e.fields.items():
        lines.append(f'  {k:<{w}} = {{{v}}},')
    lines.append('}')
    return '\n'.join(lines)


def bibtex(project, references: str = 'checks', es=None) -> str:
    from .. import __version__
    es = entries(project, references) if es is None else es
    name = getattr(project, 'name', '') if project is not None else ''
    head = [f'% BibTeX written by DexVantage {__version__}' + (f' for the project "{name}"' if name else ''),
            '% Every source the project depends on; urldate is the date each source was read.', '']
    out, group = head, None
    for e in es:
        if e.group != group:
            group = e.group
            out.append(f'% ---- {group}')
        out.append(format_entry(e))
        out.append('')
    return '\n'.join(out)


def write(project, path: str, references: str = 'checks') -> dict:
    es = entries(project, references)
    text = bibtex(project, references, es)
    if path in ('', '-'):
        import sys
        sys.stdout.write(text)
    else:
        tmp = path + '.part'
        with open(tmp, 'w', encoding='utf-8') as fh:
            fh.write(text)
        os.replace(tmp, path)
    return {'path': path, 'entries': len(es), 'keys': [e.key for e in es]}


def missing_library(project) -> list:
    """Library ids the project names that the library does not hold (they cannot be cited)."""
    return _library_entries(project)[1] if project is not None else []
