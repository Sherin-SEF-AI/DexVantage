"""What a DexVantage release is made of, in one place for the PyInstaller spec, packaging/build.py, the GitHub workflow
and tests/test_packaging.py: the package's data files, the resource folders, the version and its CHANGELOG entry, the
optional extras the bundle carries; and the release itself (artefacts, checksums, notes, the gh command).

    python packaging/release.py version                 prints the version (dexvantage/__init__.py)
    python packaging/release.py notes [--tag vX.Y.Z]    prints the CHANGELOG entry of that version (release notes);
                                                        fails when the tag is not v<version> or the entry is missing
    python packaging/release.py extras                  measures each optional extra's installed size (MB) on top of
                                                        the core requirements and the bundled extras
    python packaging/release.py [--dist DIR] [--out DIR] [--no-build] [--dry-run] [--publish]
        builds this Linux machine's bundle with packaging/build.py --smoke (unless --no-build or --dry-run), then writes
        into --out (default: --dist) SHA256SUMS.txt over every artefact in --dist and RELEASE_NOTES.md from the
        CHANGELOG, and PRINTS the gh release command. Only --publish runs that command (never from an agent or CI
        without the maintainer's say-so). --dry-run builds nothing and takes the artefacts already in --dist; a
        version without a CHANGELOG entry of its own (a .dev version) then takes the [Unreleased] entry, said in the
        notes' first line.

Ported from Vantage packaging/release.py. Reads the source files only (no import of the app), so it runs before any
requirement is installed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import re
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, 'packaging')
PKG = 'dexvantage'
REPO = 'Sherin-SEF-AI/DexVantage'
ARTEFACT_SUFFIXES = ('.tar.gz',)                 # the Linux bundles (DexVantage supports Linux only)
# files at the top of the source tree the bundle carries: the licence, the citation (core/cite.py reads CITATION.cff
# from the folder above the package, which in a bundle is the bundle's _internal folder) and the third-party record
TOP_FILES = ('LICENSE', 'CITATION.cff', 'THIRD_PARTY.md')


def version(root: str = ROOT) -> str:
    """__version__ from dexvantage/__init__.py."""
    with open(os.path.join(root, PKG, '__init__.py'), encoding='utf-8') as fh:
        m = re.search(r'''^__version__\s*=\s*['"]([^'"]+)['"]''', fh.read(), re.M)
    if not m:
        raise ValueError(f'no __version__ in {PKG}/__init__.py')
    return m.group(1)


def is_prerelease(ver: str) -> bool:
    """A development version or a pre-release (alpha, beta, release candidate): published with --prerelease."""
    return bool(re.search(r'(dev|a|b|rc)\d*$', ver))


def is_dev(ver: str) -> bool:
    """A development version (.devN): never published; a release candidate such as 0.1.0rc1 may be."""
    return bool(re.search(r'\.?dev\d*$', ver))


def extras_table(root: str = ROOT) -> dict:
    """dexvantage/extras.py EXTRAS, loaded by path (standard library only, no import of the package)."""
    path = os.path.join(root, PKG, 'extras.py')
    spec = importlib.util.spec_from_file_location('_dexvantage_extras', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod            # dataclass-like NamedTuple annotations resolve through sys.modules
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)
    return mod.EXTRAS


def resource_dirs(root: str = ROOT) -> list[str]:
    """Every subfolder of dexvantage/resources (fonts, icons, library, robots, usd_robots, ...)."""
    res = os.path.join(root, PKG, 'resources')
    return sorted(d for d in os.listdir(res) if os.path.isdir(os.path.join(res, d)) and d != '__pycache__')


def data_files(root: str = ROOT) -> list[tuple[str, str]]:
    """(source file, destination folder) for every file of the package that is not Python: the resources (fonts,
    icons, the library JSON, the bundled robots and USD robots, the schemas) and the JSON files next to modules
    (core/exporters/delta_timestamps.json, validation/VANTAGE_SLIP.json, ...); plus TOP_FILES at the bundle's top.
    The package's .py sources go in as well, by the spec's module_collection_mode (pyz+py)."""
    out = []
    pkg = os.path.join(root, PKG)
    for d, subdirs, files in os.walk(pkg):
        subdirs[:] = sorted(s for s in subdirs if s != '__pycache__' and not s.startswith('.'))
        rel = os.path.relpath(d, root)
        for f in sorted(files):
            if not f.endswith(('.py', '.pyc', '.pyo')):
                out.append((os.path.join(d, f), rel))
    out += [(os.path.join(root, f), '.') for f in TOP_FILES if os.path.isfile(os.path.join(root, f))]
    return out


_LICENCE = re.compile(r'(LICEN[CS]E|COPYING|NOTICE)', re.I)


def license_files(modules, notice_path: str) -> list[tuple[str, str]]:
    """(source file, destination folder) for the licence files of every installed distribution that provides one of
    the top-level modules given (what the bundle collected), under licenses/<distribution>/, plus Python's own and a
    notice listing each component with its version and licence, written to notice_path."""
    import importlib.metadata as md
    by_mod = md.packages_distributions()
    names = sorted({d for m in modules for d in by_mod.get(m, ())}, key=str.lower)
    out, lines = [], []
    for name in names:
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            continue
        meta = dist.metadata
        lic = meta.get('License-Expression') or (meta.get('License') or '').strip().splitlines()[0:1]
        lic = lic if isinstance(lic, str) else (lic[0] if lic else '')
        if not lic or len(lic) > 60:
            classifiers = [c.split('::')[-1].strip() for c in meta.get_all('Classifier') or []
                           if c.startswith('License ::')]
            lic = ', '.join(classifiers) or (lic[:60] + '…' if lic else 'see the files')
        found = 0
        for f in dist.files or []:
            rel = str(f).replace('\\', '/')
            if not (_LICENCE.search(f.name) or '/licenses/' in rel) or rel.endswith(('.py', '.pyc')):
                continue
            src = str(f.locate())
            if not os.path.isfile(src):
                continue
            sub = re.sub(r'^[^/]+\.dist-info/(licenses/)?', '', rel)
            out.append((src, '/'.join(['licenses', name] + ([os.path.dirname(sub)] if os.path.dirname(sub) else []))))
            found += 1
        lines.append(f'{name} {dist.version} — {lic}' + ('' if found else ' (no licence file in its package)'))
    for cand in (os.path.join(os.path.dirname(os.__file__), 'LICENSE.txt'),
                 os.path.join(sys.base_prefix, 'LICENSE.txt'),
                 os.path.join(sys.base_prefix, 'share', 'doc', 'python3', 'copyright')):
        if os.path.isfile(cand):
            out.append((cand, 'licenses/Python'))
            break
    lines.insert(0, f'Python {sys.version.split()[0]} — PSF-2.0')
    with open(notice_path, 'w', encoding='utf-8') as fh:
        fh.write(f'DexVantage {version()} bundles these open-source components. Each one\'s licence is in the folder '
                 f'of the same name next to this file;\nDexVantage\'s own source is at https://github.com/{REPO}\n\n'
                 + '\n'.join(lines) + '\n\nThe robot models, meshes and fonts DexVantage carries are listed with '
                 'their licences in THIRD_PARTY.md at the top of this folder.\nThe system libraries PyInstaller copied '
                 'from the build machine (OpenSSL, libxcb, fontconfig and the like on Linux) are under their own '
                 'licences, as their distributions publish them.\n')
    out.append((notice_path, 'licenses'))
    return out


# ------------------------------------------------------------------------------------------------ CHANGELOG
_HEAD = re.compile(r'^## +\[?v?(\d+\.\d+\.\d+[^\]\s]*|Unreleased)\]?(.*)$', re.I)


def changelog(root: str = ROOT) -> list[tuple[str, str, str]]:
    """[(version or 'Unreleased', rest of the heading line, body)] from CHANGELOG.md, newest first."""
    with open(os.path.join(root, 'CHANGELOG.md'), encoding='utf-8') as fh:
        lines = fh.read().splitlines()
    out, cur = [], None
    for ln in lines:
        m = _HEAD.match(ln)
        if m:
            cur = [m.group(1), m.group(2).strip(), []]
            out.append(cur)
        elif cur is not None:
            cur[2].append(ln)
    return [(v, head, '\n'.join(body).strip() + '\n') for v, head, body in out]


def notes(ver: str, root: str = ROOT) -> str:
    """The CHANGELOG entry of version ver (its body, without the heading)."""
    for v, _, body in changelog(root):
        if v == ver:
            return body
    raise ValueError(f'CHANGELOG.md has no entry for {ver}')


def release_notes(ver: str, root: str = ROOT, allow_unreleased: bool = False) -> str:
    """The notes of a release of ver: its CHANGELOG entry; with allow_unreleased, the [Unreleased] entry when ver has
    none (a dry run of a .dev version), said in a first line."""
    try:
        return notes(ver, root)
    except ValueError:
        if not allow_unreleased:
            raise
    for v, _, body in changelog(root):
        if v.lower() == 'unreleased':
            return (f'(Dry run: CHANGELOG.md has no entry for {ver} yet; these are its [Unreleased] notes.)\n\n'
                    + body)
    raise ValueError(f'CHANGELOG.md has no entry for {ver} and no [Unreleased] entry')


# ------------------------------------------------------------------------------------------------ extras
def measure_extras(root: str = ROOT) -> list[tuple[str, float | None, list]]:
    """[(extra, MB, [(distribution, MB)])]: the installed size of the distributions each optional extra adds on top
    of the core requirements and the bundled extras (summed over each distribution's RECORD), in this environment."""
    import importlib.metadata as md
    import tomllib
    with open(os.path.join(root, 'pyproject.toml'), 'rb') as fh:
        pp = tomllib.load(fh)

    def name_of(req: str) -> str:
        return re.sub(r'[-_.]+', '-', re.split(r'[\s;<>=!~\[(]', req.strip(), maxsplit=1)[0]).lower()

    def closure(names):
        seen, todo = set(), list(names)
        while todo:
            n = name_of(todo.pop())
            if n in seen:
                continue
            try:
                d = md.distribution(n)
            except md.PackageNotFoundError:
                continue
            seen.add(n)
            for r in d.requires or []:
                if ';' in r and 'extra ==' in r.split(';', 1)[1]:
                    continue                       # the dependency's own extras
                todo.append(r)
        return seen

    def size(n):
        total = 0
        for f in md.distribution(n).files or []:
            try:
                total += os.path.getsize(f.locate())
            except OSError:
                pass
        return total / 1e6

    table = extras_table(root)
    opt = pp['project'].get('optional-dependencies', {})
    core = closure(pp['project']['dependencies'])
    bundle = set(core)
    for name, e in table.items():
        if e.bundled:
            bundle |= closure(opt.get(name, ()))
    out = []
    for name in opt:
        if name not in table:
            continue
        own = closure(opt[name])
        if not own:
            out.append((name, None, []))
            continue
        # a bundled extra: what it adds to the core; a left-out one: what it would add to the bundle as built
        parts = [(n, size(n)) for n in sorted(own - (core if table[name].bundled else bundle))]
        out.append((name, sum(s for _, s in parts), parts))
    return out


# ------------------------------------------------------------------------------------------------ release
def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def artefacts(dist: str) -> list[str]:
    """The release files in dist (archives and installers), sorted."""
    if not os.path.isdir(dist):
        return []
    return sorted(os.path.join(dist, f) for f in os.listdir(dist)
                  if f.endswith(ARTEFACT_SUFFIXES) and os.path.isfile(os.path.join(dist, f)))


def write_checksums(files, out_dir: str) -> str:
    """SHA256SUMS.txt in out_dir, in sha256sum's format (sha256sum -c reads it in the artefacts' folder)."""
    path = os.path.join(out_dir, 'SHA256SUMS.txt')
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        for f in files:
            fh.write(f'{sha256(f)}  {os.path.basename(f)}\n')
    return path


def gh_command(ver: str, files, notes_path: str) -> list[str]:
    return ['gh', 'release', 'create', f'v{ver}', *files, '--repo', REPO, '--verify-tag',
            '--title', f'DexVantage {ver}', '--notes-file', notes_path] + (['--prerelease'] if is_prerelease(ver)
                                                                             else [])


def release(dist: str, out: str, build: bool = True, dry_run: bool = False, publish: bool = False,
            root: str = ROOT) -> int:
    ver = version(root)
    if publish and dry_run:
        print('release.py: --publish and --dry-run together make no sense; give one.', file=sys.stderr)
        return 2
    if publish and is_dev(ver):
        print(f'release.py: {ver} is a development version; set the release version in {PKG}/__init__.py and '
              'give it a CHANGELOG entry before publishing.', file=sys.stderr)
        return 1
    if build and not dry_run:
        r = subprocess.run([sys.executable, os.path.join(HERE, 'build.py'), '--smoke', '--dist', dist])
        if r.returncode:
            print(f'release.py: the build or its smoke test failed (exit {r.returncode}); nothing to release.',
                  file=sys.stderr)
            return 1
    files = artefacts(dist)
    if not files and not dry_run:
        print(f'release.py: no release files ({", ".join(ARTEFACT_SUFFIXES)}) in {dist}.', file=sys.stderr)
        return 1
    try:
        text = release_notes(ver, root, allow_unreleased=dry_run)
    except ValueError as e:
        print(f'release.py: {e}.', file=sys.stderr)
        return 1
    os.makedirs(out, exist_ok=True)
    sums = write_checksums(files, out)
    notes_path = os.path.join(out, 'RELEASE_NOTES.md')
    with open(notes_path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(text)
    cmd = gh_command(ver, files + [sums], notes_path)
    print(f'{len(files)} release file{"" if len(files) == 1 else "s"}' + (' (dry run: nothing built)' if dry_run
                                                                           else ''))
    with open(sums, encoding='utf-8') as fh:
        sys.stdout.write(fh.read())
    print(f'release notes: {notes_path}')
    if not publish:
        print('Not published. To publish this release (after tagging v' + ver + ' and pushing the tag), run:')
        print('  ' + ' '.join(shlex.quote(c) for c in cmd))
        return 0
    print('+ ' + ' '.join(shlex.quote(c) for c in cmd), flush=True)
    return subprocess.run(cmd).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd')
    sub.add_parser('version')
    n = sub.add_parser('notes')
    n.add_argument('--tag', default='', help='the git tag being released; must be v<version>')
    sub.add_parser('extras')
    ap.add_argument('--dist', default=os.path.join(ROOT, 'out', 'dist'), help='where the artefacts are built')
    ap.add_argument('--out', default='', help='where SHA256SUMS.txt and RELEASE_NOTES.md go (default: --dist)')
    ap.add_argument('--no-build', action='store_true', help='release the artefacts already in --dist')
    ap.add_argument('--dry-run', action='store_true', help='build nothing; checksums and notes of what is in --dist')
    ap.add_argument('--publish', action='store_true', help='run the gh release command (the maintainer only)')
    args = ap.parse_args(argv)
    ver = version()
    if args.cmd == 'version':
        print(ver)
        return 0
    if args.cmd == 'extras':
        for name, mb, parts in measure_extras():
            print(f'{name:<10} ' + ('not installed' if mb is None else f'{mb:7.1f} MB  ')
                  + ', '.join(f'{p} {s:.1f}' for p, s in parts))
        return 0
    if args.cmd == 'notes':
        if args.tag and args.tag != f'v{ver}':
            print(f'release.py: tag {args.tag} does not match the version {ver} in {PKG}/__init__.py (tag v{ver})',
                  file=sys.stderr)
            return 1
        entries = [e for e in changelog() if e[0].lower() != 'unreleased']
        if not entries or entries[0][0] != ver:
            print(f'release.py: the newest released entry of CHANGELOG.md is '
                  f'{entries[0][0] if entries else "missing"}, not {ver}', file=sys.stderr)
            return 1
        sys.stdout.buffer.write(notes(ver).encode('utf-8'))     # UTF-8 whatever the console's code page
        sys.stdout.flush()
        return 0
    dist = os.path.abspath(args.dist)
    return release(dist, os.path.abspath(args.out or dist), build=not args.no_build, dry_run=args.dry_run,
                   publish=args.publish)


if __name__ == '__main__':
    sys.exit(main())
