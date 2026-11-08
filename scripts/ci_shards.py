"""Split the test files into CI shards of about equal running time (.github/workflows/ci.yml).

    python scripts/ci_shards.py --shards 6 --index 2              print shard 2's test files, one per line
    python scripts/ci_shards.py --shards 6 --index 2 --run -- -q  run pytest on them (arguments after -- go to pytest)
    python scripts/ci_shards.py --shards 6 --list                 every shard with its files and estimated seconds

A shard is a list of whole files: module-scope fixtures (samples, windows, rendered episodes) stay within one process,
and each CI job starts a fresh process, so no job carries the memory the whole suite builds up (the full suite's
process grows to about 5 GB, and its cgroup peaked at 6.4 GB with its render workers: docs/BUILD_STATE.md).

The weights are measured seconds per file (scripts/ci_test_durations.json, from one full junit run); a file not in
the table counts as the median. SOLO files get a shard of their own, because of their memory, not their time:
test_validation's quick run of every check runs up to two check models in child processes at once, about 6 GB
together (tests/conftest.py). The other files go, longest first, to the shard with the least time so far (the LPT
rule); ties go to the lower shard number, so the split is the same on every machine.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DURATIONS = os.path.join(ROOT, 'scripts', 'ci_test_durations.json')
SOLO = ('test_validation.py',)


def test_files(root: str = ROOT) -> list[str]:
    """Every test module under tests/ (file names, sorted)."""
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, 'tests', 'test_*.py')))


def weights(files: list[str], path: str = DURATIONS) -> dict[str, float]:
    """Measured seconds per file; the median of the measured ones for a file the table does not list."""
    try:
        with open(path, encoding='utf-8') as f:
            table = json.load(f).get('seconds', {})
    except (OSError, ValueError):
        table = {}
    known = [float(v) for k, v in table.items() if k in files]
    median = statistics.median(known) if known else 1.0
    return {f: float(table.get(f, median)) for f in files}


def split(files: list[str], n: int, w: dict[str, float] | None = None, solo=SOLO) -> list[list[str]]:
    """n shards (lists of file names) covering every file once. Each SOLO file present takes a shard of its own
    (when n leaves at least one shard for the rest); the others are spread by the LPT rule."""
    if n < 1:
        raise ValueError('the number of shards must be at least 1')
    w = w if w is not None else weights(files)
    shards: list[list[str]] = [[] for _ in range(n)]
    alone = [f for f in files if f in solo] if n > 1 else []
    alone = alone[:n - 1]
    for i, f in enumerate(alone):
        shards[i].append(f)
    free = list(range(len(alone), n))
    load = {i: 0.0 for i in free}
    for f in sorted((f for f in files if f not in alone), key=lambda f: (-w.get(f, 0.0), f)):
        i = min(free, key=lambda k: (load[k], k))
        shards[i].append(f)
        load[i] += w.get(f, 0.0)
    return [sorted(s) for s in shards]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Split the test files into CI shards of about equal running time.')
    ap.add_argument('--shards', type=int, required=True, help='how many shards')
    ap.add_argument('--index', type=int, help='which shard (0 to shards − 1)')
    ap.add_argument('--list', action='store_true', help='print every shard with its estimated seconds')
    ap.add_argument('--run', action='store_true', help='run pytest on the shard (arguments after -- go to pytest)')
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if '--' in argv:
        k = argv.index('--')
        argv, extra = argv[:k], argv[k + 1:]
    a = ap.parse_args(argv)
    if a.shards < 1:
        print('The number of shards must be at least 1.', file=sys.stderr)
        return 2
    files = test_files()
    w = weights(files)
    shards = split(files, a.shards, w)
    if a.list:
        for i, s in enumerate(shards):
            print(f'shard {i}: {len(s)} files, about {sum(w[f] for f in s):.0f} s (measured on the reference machine)')
            for f in s:
                print(f'    {f}')
        return 0
    if a.index is None or not 0 <= a.index < a.shards:
        print(f'Give --index between 0 and {a.shards - 1} (or --list).', file=sys.stderr)
        return 2
    mine = [os.path.join('tests', f) for f in shards[a.index]]
    if not a.run:
        print('\n'.join(mine))
        return 0
    if not mine:
        print(f'Shard {a.index} of {a.shards} has no test files.')
        return 0
    print(f'Shard {a.index} of {a.shards}: {len(mine)} files, about {sum(w[f] for f in shards[a.index]):.0f} s on '
          'the reference machine.', flush=True)
    return subprocess.call([sys.executable, '-m', 'pytest', *extra, *mine], cwd=ROOT)


if __name__ == '__main__':
    sys.exit(main())
