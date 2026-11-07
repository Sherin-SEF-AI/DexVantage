"""Spec §1.4: no personal paths, e-mail addresses or keys in the files that go into the repository."""
import os
import re
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
PATTERNS = [re.compile(p) for p in (r'/home/[a-z]', r'/Users/[A-Za-z]', r'/tmp/claude', r'sk-ant-[A-Za-z0-9]',
                                    r'gsk_[A-Za-z0-9]{10}', r'[A-Za-z0-9._%+-]+@gmail\.com')]
SKIP_DIRS = {'.git', '.venv', 'out', '__pycache__', '.pytest_cache', '.ruff_cache', 'build', 'dist'}


def _files():
    try:
        out = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return [os.path.join(ROOT, f) for f in out.stdout.splitlines()]
    except OSError:
        pass
    found = []
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.endswith('.egg-info')]
        found += [os.path.join(root, f) for f in files]
    return found


def test_no_personal_paths_or_secrets():
    hits = []
    for path in _files():
        if path.endswith(('.png', '.ttf', '.dexv', '.npz')) or not os.path.isfile(path):
            continue
        if os.path.basename(path) == 'test_hygiene.py':
            continue
        text = open(path, encoding='utf-8', errors='ignore').read()
        for rx in PATTERNS:
            for m in rx.finditer(text):
                hits.append(f'{os.path.relpath(path, ROOT)}: {m.group(0)}')
    assert not hits, hits


def test_no_large_files():
    """No file over 2 MB, except a bundled robot model's mesh, which may reach 5 MB (the samples' package budget: no
    file over 5 MB, at most about 12 MB per robot, tests/test_samples.py; ALOHA 2's wrist-mount mesh is 2.3 MB)."""
    robots = os.path.join(ROOT, 'dexvantage', 'resources', 'robots') + os.sep

    def limit(p):
        return 5_000_000 if p.startswith(robots) else 2_000_000
    big = [os.path.relpath(p, ROOT) for p in _files() if os.path.isfile(p) and os.path.getsize(p) > limit(p)]
    assert not big, big
