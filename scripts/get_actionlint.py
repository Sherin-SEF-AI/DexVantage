"""Fetch actionlint (the GitHub Actions workflow linter, github.com/rhysd/actionlint, MIT) at a pinned version into
out/tools/actionlint-<version>/, checked against the SHA-256 its release publishes, and print the binary's path.

    python scripts/get_actionlint.py            download (once) and print the path
    python scripts/get_actionlint.py --path     print where it would be, without downloading

The checksums below are copied from actionlint_1.7.12_checksums.txt of the v1.7.12 release (published 2026-03-30,
read 2026-11-08); a download whose SHA-256 differs is deleted and refused. tests/test_ci_workflows.py runs the binary
when it is there and otherwise checks the workflows with PyYAML alone. Nothing is installed outside out/tools.
DexVantage supports Linux only, so only the two Linux builds (x86_64 and arm64) are pinned.
"""
from __future__ import annotations

import hashlib
import io
import os
import platform
import stat
import sys
import tarfile
import urllib.request

VERSION = '1.7.12'
URL = 'https://github.com/rhysd/actionlint/releases/download/v{v}/{name}'
SHA256 = {
    'linux_amd64': '8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8',
    'linux_arm64': '325e971b6ba9bfa504672e29be93c24981eeb1c07576d730e9f7c8805afff0c6',
}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def target() -> str:
    """The release's platform key for this machine, e.g. 'linux_amd64'."""
    osname = 'linux' if sys.platform.startswith('linux') else sys.platform
    m = platform.machine().lower()
    arch = {'x86_64': 'amd64', 'amd64': 'amd64', 'aarch64': 'arm64', 'arm64': 'arm64'}.get(m, m)
    return f'{osname}_{arch}'


def binary_path(root: str = ROOT) -> str:
    return os.path.join(root, 'out', 'tools', f'actionlint-{VERSION}', 'actionlint')


def fetch(root: str = ROOT) -> str:
    """The binary's path, downloading and verifying it first when it is not there yet. Raises RuntimeError."""
    path = binary_path(root)
    if os.path.isfile(path):
        return path
    key = target()
    if key not in SHA256:
        raise RuntimeError(f'actionlint {VERSION} has no pinned checksum for {key}; run it from PATH instead.')
    name = f'actionlint_{VERSION}_{key}.tar.gz'
    url = URL.format(v=VERSION, name=name)
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
    except OSError as e:
        raise RuntimeError(f'Could not download {url}: {e}') from None
    got = hashlib.sha256(data).hexdigest()
    if got != SHA256[key]:
        raise RuntimeError(f'{name} has SHA-256 {got}, not the published {SHA256[key]}: refused.')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    member = os.path.basename(path)
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as t:
        f = t.extractfile(member)
        if f is None:
            raise RuntimeError(f'{name} has no {member} in it.')
        blob = f.read()
    tmp = path + '.part'
    with open(tmp, 'wb') as f:
        f.write(blob)
    os.chmod(tmp, os.stat(tmp).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    os.replace(tmp, path)
    return path


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if '--path' in argv:
        print(binary_path())
        return 0
    try:
        print(fetch())
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
