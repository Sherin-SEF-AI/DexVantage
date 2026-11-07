"""Slurm job arrays: a job folder that runs a work's shards as the tasks of one array.

    SETTINGS / DEFAULTS                    the sbatch settings, each a labelled DexVantage default unless set
    read_settings(given) -> dict           checked settings (SlurmError in a sentence for one that cannot be used)
    sbatch_script(settings, shards, ...)   the array script's text
    directives(text) -> [(option, value)]  the #SBATCH lines Slurm reads (up to the first command line)
    write_job_folder(work, folder, ...)    work.json, projects/, run.sbatch, job.json, out/, logs/, cache/, README.txt
    write_resubmit(folder, shards, of)     resubmit.sbatch: the same script for just those array indices
    submit(folder, script, dependency)     sbatch --parsable [--dependency=afterany:ID] script; the job id

Each array task runs `python -m dexvantage --run-shard work.json --shard $SLURM_ARRAY_TASK_ID --of N --out out` in the
job folder; `dexvantage --gather FOLDER` then merges the shards. Slurm's facts used here, from its documentation
(https://slurm.schedmd.com/job_array.html and https://slurm.schedmd.com/sbatch.html, version 26.05, read 2026-11-07):
--array takes ranges and comma lists ("--array=1-7:2", "--array=1,3,5,7") and a "%" throttle on the tasks running at
once ("--array=0-15%4"); SLURM_ARRAY_TASK_ID is "the job array index value"; %A and %a in --output are the array's
job id and the task's index; "The default value of MaxArraySize is 1001" (indices 0 to 1000), so a folder of more
shards than that is refused unless max_array says the cluster allows more; --mem's "Default units are mebibytes" with
the suffixes K, M, G, T; --time takes "minutes", "minutes:seconds", "hours:minutes:seconds" and the day forms
"days-hours", "days-hours:minutes", "days-hours:minutes:seconds"; without --cpus-per-task "the controller will just
try to allocate one processor per task"; SLURM_SUBMIT_DIR is "The directory from which sbatch was invoked"; "Once the
first non-comment, non-whitespace line has been reached in the script, no more #SBATCH directives will be processed";
--parsable "Outputs only the job ID number and the cluster name if present", separated by a semicolon; afterany
starts a job "after the specified jobs have terminated".

The defaults (labelled defaults, not Slurm's and not a site's): one CPU per task, 4G of memory (above the largest
worker peak measured on the samples, docs/models/cluster.md), 2 hours, the cluster's default partition and account,
no GPU, no throttle, `python` from the PATH after the activation line. The activation line has no default path: it is
the user's (for example `source <your venv>/bin/activate` or `module load python`), written into the script as given.

Known limits: PBS, LSF and other schedulers are out of scope (only Slurm job arrays are written); the script does not
stage files to node-local disk (the job folder must be on a filesystem every node mounts); the time and memory a task
needs are not predicted (set them from a small run: gather reports the shards' peak memory); a site that names its
GPUs (--gpus=a100:1) needs the extra-lines setting.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time

from .work import ClusterError, Work

JOB_SCHEMA = 'dexvantage.cluster-job/1'
MAX_ARRAY_DEFAULT = 1001             # Slurm's default MaxArraySize (job_array.html): indices 0 … 1000
DEFAULTS = {'cpus_per_task': 1, 'mem': '4G', 'time': '02:00:00', 'partition': '', 'account': '',
            'job_name': 'dexvantage', 'gpus': 0, 'throttle': 0, 'activate': '', 'setup': [], 'python': 'python',
            'extra': [], 'max_array': MAX_ARRAY_DEFAULT, 'threads_from_cpus': True}
SETTINGS = {
    'cpus_per_task': 'CPUs per task (#SBATCH --cpus-per-task; default 1)',
    'mem': 'memory per task (#SBATCH --mem; default 4G, a DexVantage default above the measured worker peaks)',
    'time': 'time limit per task (#SBATCH --time; default 02:00:00)',
    'partition': 'partition (#SBATCH --partition; default: the cluster\'s own)',
    'account': 'account to charge (#SBATCH --account; default: the user\'s own)',
    'job_name': 'job name (#SBATCH --job-name; default dexvantage)',
    'gpus': 'GPUs per task (#SBATCH --gpus; default 0: DexVantage then uses its CPU paths)',
    'throttle': 'tasks running at once (the --array "%" throttle; default 0: no limit)',
    'activate': 'the line that activates DexVantage\'s environment, e.g. source <your venv>/bin/activate (no '
                'default)',
    'setup': 'shell lines run before it, e.g. module load python (default none)',
    'python': 'the Python command (default python)',
    'extra': 'further #SBATCH options, each starting with -- (default none)',
    'max_array': 'the cluster\'s MaxArraySize (default 1001, Slurm\'s default)',
    'threads_from_cpus': 'set OMP, OpenBLAS and MKL threads to the CPUs per task (default on)',
}
_NAME = re.compile(r'^[A-Za-z0-9_.+\-]+$')
_MEM = re.compile(r'^\d+[KMGT]?$')
_TIME = re.compile(r'^(\d+|\d+:\d{1,2}|\d+:\d{1,2}:\d{1,2}|\d+-\d+|\d+-\d+:\d{1,2}|\d+-\d+:\d{1,2}:\d{1,2})$')


class SlurmError(ClusterError):
    """A Slurm setting or job folder that cannot be used; the message says why."""


def read_settings(given: dict | None = None) -> dict:
    """The settings with every key not given at its default; SlurmError for a value Slurm would refuse."""
    given = dict(given or {})
    unknown = sorted(set(given) - set(DEFAULTS))
    if unknown:
        raise SlurmError(f'Unknown Slurm setting {", ".join(unknown)}; the settings are {", ".join(DEFAULTS)}.')
    st = {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULTS.items()}
    st.update(given)
    st['defaults'] = sorted(k for k in DEFAULTS if k not in given)
    for k in ('cpus_per_task', 'gpus', 'throttle', 'max_array'):
        try:
            st[k] = int(st[k])
        except (TypeError, ValueError):
            raise SlurmError(f'The Slurm setting {k} must be a whole number; it is {st[k]!r}.') from None
        if st[k] < (1 if k in ('cpus_per_task', 'max_array') else 0):
            raise SlurmError(f'The Slurm setting {k} must be {"at least 1" if k != "gpus" else "0 or more"}; it is '
                             f'{st[k]}.')
    st['mem'] = str(st['mem']).strip().upper()
    if not _MEM.match(st['mem']):
        raise SlurmError(f'The memory per task "{st["mem"]}" is not a Slurm size: a number of mebibytes or a number '
                         'with K, M, G or T (for example 4G).')
    st['time'] = str(st['time']).strip()
    if not _TIME.match(st['time']):
        raise SlurmError(f'The time limit "{st["time"]}" is not a Slurm time: minutes, minutes:seconds, '
                         'hours:minutes:seconds or days-hours[:minutes[:seconds]] (for example 02:00:00).')
    for k in ('partition', 'account', 'job_name'):
        st[k] = str(st[k] or '').strip()
        if st[k] and not _NAME.match(st[k]):
            raise SlurmError(f'The Slurm {k.replace("_", " ")} "{st[k]}" may hold only letters, digits and . _ + -.')
    if not st['job_name']:
        st['job_name'] = DEFAULTS['job_name']
    for k in ('activate', 'python'):
        st[k] = str(st[k] or '').strip()
        if '\n' in st[k] or '\r' in st[k]:
            raise SlurmError(f'The Slurm setting {k} must be one line.')
    if not st['python']:
        st['python'] = DEFAULTS['python']
    for k in ('setup', 'extra'):
        v = st[k]
        if isinstance(v, str):
            v = [x for x in v.splitlines() if x.strip()]
        if not isinstance(v, list) or any(not isinstance(x, str) or '\n' in x for x in v):
            raise SlurmError(f'The Slurm setting {k} must be a list of lines.')
        st[k] = [x.strip() for x in v if x.strip()]
    for x in st['extra']:
        if not x.startswith('--'):
            raise SlurmError(f'The extra #SBATCH option "{x}" must start with -- (for example --qos=short).')
    st['threads_from_cpus'] = bool(st['threads_from_cpus'])
    return st


def directives(text: str) -> list:
    """[(option, value or None)] of the #SBATCH lines, read as Slurm reads them: up to the first line that is neither
    a comment nor blank."""
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if not s.startswith('#'):
            break
        if s.startswith('#SBATCH'):
            for tok in shlex.split(s[len('#SBATCH'):].strip(), comments=False):
                if tok.startswith('#'):
                    break
                if '=' in tok:
                    k, v = tok.split('=', 1)
                    out.append((k, v))
                else:
                    out.append((tok, None))
    return out


def sbatch_script(settings: dict, shards: int, array: str | None = None, title: str = '', work_file='work.json',
                  out_dir='out', items: int | None = None) -> str:
    """The text of the array script. array: the --array indices ('0-N−1' by default; a resubmit's list)."""
    from ... import __version__
    st = settings if 'defaults' in settings else read_settings(settings)
    shards = int(shards)
    arr = array or (f'0-{shards - 1}' if shards > 1 else '0')
    if st['throttle']:
        arr += f'%{st["throttle"]}'
    lines = ['#!/bin/bash',
             f'#SBATCH --job-name={st["job_name"]}',
             f'#SBATCH --array={arr}',
             f'#SBATCH --cpus-per-task={st["cpus_per_task"]}',
             f'#SBATCH --mem={st["mem"]}',
             f'#SBATCH --time={st["time"]}',
             '#SBATCH --output=logs/shard-%A_%a.out']
    if st['partition']:
        lines.append(f'#SBATCH --partition={st["partition"]}')
    if st['account']:
        lines.append(f'#SBATCH --account={st["account"]}')
    if st['gpus']:
        lines.append(f'#SBATCH --gpus={st["gpus"]}')
    lines += [f'#SBATCH {x}' for x in st['extra']]
    defaults = [k for k in st['defaults'] if k in ('cpus_per_task', 'mem', 'time', 'gpus', 'throttle')]
    lines += ['#',
              f'# DexVantage {__version__}: {title or "work"}' + (f', {items} items' if items is not None else '') +
              f' in {shards} shard{"" if shards == 1 else "s"}.',
              '# Submit from this folder: sbatch ' + ('run.sbatch' if array is None else 'resubmit.sbatch') +
              '; then gather: python -m dexvantage --gather .',
              '# ' + (f'DexVantage defaults (labelled, not your site\'s): {", ".join(defaults)}.' if defaults else
                      'Every resource is set as given.'),
              '',
              'set -euo pipefail',
              'JOB_DIR="${DEXVANTAGE_JOB_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"',
              'cd "$JOB_DIR"']
    lines += st['setup']
    if st['activate']:
        lines.append(st['activate'])
    if st['threads_from_cpus']:
        c = st['cpus_per_task']
        lines.append(' '.join(['export'] + [f'{v}="${{SLURM_CPUS_PER_TASK:-{c}}}"' for v in
                                            ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')]))
    lines += ['SHARD="${SLURM_ARRAY_TASK_ID:?run this script with sbatch: it is a Slurm array task}"',
              f'exec {st["python"]} -m dexvantage --run-shard {shlex.quote(work_file)} --shard "$SHARD" --of {shards} '
              f'--out {shlex.quote(out_dir)}',
              '']
    return '\n'.join(lines)


def _readme(work: Work, shards: int, st: dict) -> str:
    return '\n'.join([
        f'DexVantage job folder: {work.describe()} in {shards} shards.',
        '',
        'On the cluster, from this folder:',
        '    sbatch run.sbatch                    one array task per shard (logs/ holds each task\'s output)',
        '    python -m dexvantage --gather .      merge the shards into result.json, or say which shards are',
        '                                         missing or failed and write resubmit.sbatch for just those',
        '',
        'Without Slurm, any shard runs by hand:',
        f'    python -m dexvantage --run-shard work.json --shard I --of {shards} --out out',
        '',
        'Settings used (job.json lists them; those in "defaults" are DexVantage defaults, not your site\'s): '
        + ', '.join(f'{k}={st[k]}' for k in ('cpus_per_task', 'mem', 'time', 'gpus', 'throttle')) + '.',
        'The folder must be on a filesystem every node mounts (out/ and cache/ are written by every task).',
        ''])


def write_job_folder(work: Work, folder: str, settings: dict | None = None, shards: int | None = None,
                     shared_cache: bool = True) -> dict:
    """Write the job folder (a new or empty folder). Returns {'folder', 'script', 'work', 'shards', 'items',
    'settings', 'text'}."""
    st = read_settings(settings)
    m = len(work.items)
    shards = int(shards) if shards else min(m, 32)
    if shards < 1:
        raise SlurmError('The job needs at least one shard.')
    if shards > m:
        shards = m
    if shards > st['max_array']:
        raise SlurmError(f'{shards} shards is more than the cluster\'s MaxArraySize of {st["max_array"]} (Slurm\'s '
                         'default is 1001): use fewer shards, or set max_array to your cluster\'s value.')
    if os.path.exists(folder) and (not os.path.isdir(folder) or os.listdir(folder)):
        raise SlurmError(f'The folder {os.path.basename(os.path.normpath(folder))} is not empty: choose a new or '
                         'empty folder for the job.')
    try:
        for sub in ('out', 'logs') + (('cache',) if shared_cache else ()):
            os.makedirs(os.path.join(folder, sub), exist_ok=True)
        wp = work.save(folder)
        script = sbatch_script(st, shards, title=work.describe(), items=m)
        sp = os.path.join(folder, 'run.sbatch')
        with open(sp, 'w', encoding='utf-8', newline='\n') as fh:
            fh.write(script)
        os.chmod(sp, 0o755)
        job = {'schema': JOB_SCHEMA, 'kind': work.kind, 'work_id': work.work_id, 'shards': shards, 'items': m,
               'slurm': {k: v for k, v in st.items() if k != 'defaults'}, 'defaults': st['defaults'],
               'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        with open(os.path.join(folder, 'job.json'), 'w', encoding='utf-8') as fh:
            json.dump(job, fh, indent=1)
            fh.write('\n')
        with open(os.path.join(folder, 'README.txt'), 'w', encoding='utf-8') as fh:
            fh.write(_readme(work, shards, st))
    except OSError as e:
        raise SlurmError(f'Could not write the job folder: {e.strerror or e}. Choose a folder you can write to.') \
            from None
    text = (f'{work.describe()} in {shards} shards → {os.path.basename(os.path.normpath(folder))}/. On the cluster, '
            'from that folder: sbatch run.sbatch, then python -m dexvantage --gather .')
    return {'folder': folder, 'script': sp, 'work': wp, 'shards': shards, 'items': m, 'settings': st, 'text': text}


def write_resubmit(folder: str, redo: list, of: int) -> str | None:
    """resubmit.sbatch in a job folder: the array script for just the shards in redo (None: not a Slurm folder)."""
    from .shards import _ranges
    try:
        with open(os.path.join(folder, 'job.json'), encoding='utf-8') as fh:
            job = json.load(fh)
    except (OSError, ValueError):
        return None
    st = read_settings({k: v for k, v in (job.get('slurm') or {}).items() if k in DEFAULTS and
                        k not in (job.get('defaults') or [])})
    text = sbatch_script(st, int(of), array=_ranges(redo), title=f'resubmit of {len(redo)} shards')
    path = os.path.join(folder, 'resubmit.sbatch')
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(text)
    os.chmod(path, 0o755)
    return path


def submit(folder: str, script: str = 'run.sbatch', dependency: str | None = None, sbatch: str = 'sbatch') -> str:
    """Submit a script of the job folder with sbatch (run in the folder); the job id. SlurmError in a sentence when
    sbatch is not found or refuses."""
    exe = shutil.which(sbatch)
    if exe is None:
        raise SlurmError('sbatch is not on the PATH: run this on a machine of the cluster (a login node), or submit '
                         f'{script} there yourself with sbatch.')
    cmd = [exe, '--parsable'] + ([f'--dependency=afterany:{dependency}'] if dependency else []) + [script]
    try:
        r = subprocess.run(cmd, cwd=folder, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SlurmError(f'sbatch could not be run ({e}).') from None
    if r.returncode != 0:
        why = (r.stderr.strip().splitlines() or ['no message'])[-1]
        raise SlurmError(f'sbatch refused {script}: {why}')
    jid = (r.stdout.strip().splitlines() or [''])[-1].split(';')[0].strip()
    if not jid:
        raise SlurmError(f'sbatch accepted {script} but printed no job id.')
    return jid
