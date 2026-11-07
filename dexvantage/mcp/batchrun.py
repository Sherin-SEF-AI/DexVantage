"""run_batch: Claude runs DexVantage's batch jobs on the project as it is now (an MCP tool, dexvantage.mcp.service).

The project (the window's or the file's, unsaved changes included) is saved to a new run folder, and batch mode
(dexvantage.batch: the same code as `dexvantage project.dexv --coverage out.json …`) runs on that copy in a process
of its own, in that folder, so relative outputs land there. The reply: the exit code and what it means, the end of the
output, the files written and the content of the small JSON ones. The project itself is not changed.

Not allowed here: --sample (the run is always on this project) and --self-test (not a job on the project).

Ported from Vantage mcp/batchrun.py (the run folder, the option check against batch.add_arguments, the reply); the
options and exit codes are DexVantage's (batch.py: 0 done, 1 a failed check or a result that does not reproduce,
2 an error that stopped the run).

Known limits: the run waits for the process (at most timeout_s, 600 s by default, 3600 s at most); outputs given as
absolute paths land where they say, outside the run folder.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

DENY = {'--sample': 'the run is always on this project', '--self-test': 'not a job on the project',
        '--version': 'not a job on the project'}
EXIT = {0: 'done', 1: 'done: a check failed or a result does not reproduce (see the output)',
        2: 'stopped: an error (a file that cannot be read, a bad option, a study that refused; see the output)',
        -9: 'stopped: it ran longer than timeout_s'}
JSON_MAX = 60_000
TAIL = 4000


def _options() -> dict:
    """option -> (takes, help) for every batch option Claude may use."""
    import argparse

    from ..batch import add_arguments
    ap = argparse.ArgumentParser(add_help=False)
    add_arguments(ap)
    out = {}
    for a in ap._actions:                               # noqa: SLF001 — argparse keeps them only here
        for o in a.option_strings:
            if o.startswith('--') and o not in DENY:
                takes = 'flag' if a.nargs == 0 else (a.metavar or a.dest.upper())
                out[o] = (takes, (a.help or '').replace('%%', '%'))
    return out


def help_doc() -> dict:
    return {'options': {o: (f'{t}: {h}' if t != 'flag' else f'(flag) {h}') for o, (t, h) in sorted(_options().items())},
            'not_allowed': DENY,
            'note': 'args are the command-line words after the project, e.g. ["--coverage", "coverage.json"]. '
                    'Relative output paths land in the run folder; several jobs can go in one run.'}


def check(args) -> str:
    """'' when args are batch options Claude may use, else why not."""
    if not isinstance(args, list) or not args or not all(isinstance(a, (str, int, float)) for a in args):
        return 'args is a list of the command-line words (strings), e.g. ["--coverage", "coverage.json"].'
    allowed = _options()
    for a in map(str, args):
        if a.startswith('--'):
            name = a.split('=', 1)[0]
            if name in DENY:
                return f'{name} is not available here: {DENY[name]}.'
            if name not in allowed:
                return f'{name} is not a batch option (run_batch with help: true lists them).'
    if not any(str(a).startswith('--') for a in args):
        return 'args name no job (an option such as --coverage OUT.json).'
    return ''


def command(project_file: str, args) -> tuple[list, dict]:
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env['PYTHONPATH'] = root + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    return [sys.executable, '-m', 'dexvantage', project_file] + [str(a) for a in args], env


def run(project, args, base_dir: str, timeout_s=600) -> dict:
    """Run the batch job on a copy of project in a new folder under base_dir."""
    why = check(args)
    if why:
        return {'error': why}
    stamp = time.strftime('%Y%m%d-%H%M%S')
    folder = os.path.join(base_dir, f'claude-run-{stamp}')
    n = 2
    while os.path.exists(folder):
        folder = os.path.join(base_dir, f'claude-run-{stamp}-{n}')
        n += 1
    os.makedirs(folder)
    pfile = os.path.join(folder, 'project.dexv')
    snap = project.snapshot()
    snap.save(pfile)                                    # the user's project (its file) is not touched
    cmd, env = command(pfile, args)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=folder, env=env, capture_output=True, text=True, errors='replace',
                           timeout=max(10, min(int(timeout_s or 600), 3600)))
        code, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        code = -9
        out = e.stdout.decode('utf-8', 'replace') if isinstance(e.stdout, bytes) else (e.stdout or '')
        err = e.stderr.decode('utf-8', 'replace') if isinstance(e.stderr, bytes) else (e.stderr or '')
    files, budget, docs = [], JSON_MAX, {}
    for root, _, names in os.walk(folder):
        for nm in sorted(names):
            f = os.path.join(root, nm)
            if f == pfile:
                continue
            rel = os.path.relpath(f, folder)
            size = os.path.getsize(f)
            files.append({'file': rel, 'bytes': size})
            if nm.endswith('.json') and size <= budget:
                try:
                    with open(f, encoding='utf-8') as fh:
                        docs[rel] = json.load(fh)
                    budget -= size
                except (OSError, ValueError):
                    pass
    return {'exit_code': code, 'meaning': EXIT.get(code, 'finished with an unusual code: see the output'),
            'seconds': round(time.time() - t0, 1), 'folder': folder, 'files': files[:200], 'json': docs,
            'stdout_end': out[-TAIL:], 'stderr_end': err[-TAIL:],
            'note': 'The run used a copy of the project; the project is unchanged.'}
