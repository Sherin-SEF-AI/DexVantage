"""Cluster sweeps (core/cluster, batch_cluster.py, dv.cluster): shards, gather, the Slurm job folder, the shared disk
cache, the local process pool and the optimiser's generations.

No Slurm is needed: sbatch is a mock script put on PATH in a temporary folder. Shards that must run "in separate
processes" run as `python -m dexvantage --run-shard ...` subprocesses. The figures come from the SO-100 sample's
coverage study (about 0.05–0.3 s per evaluation), so the comparisons are with real studies.
"""
import json
import os
import shutil
import stat
import subprocess
import sys

import pytest

from dexvantage.core import cluster as C
from dexvantage.core import experiments as EX
from dexvantage.core import jobs
from dexvantage.core.cluster import cache as CA
from dexvantage.core.cluster import shards as SH
from dexvantage.core.cluster import slurm as SL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAM = 'sensor:Third-person camera (sample placement)'
HOW = ('execution', 'seconds', 'cached', 'schema', 'study')


@pytest.fixture(autouse=True)
def _cold_occlusion_cache():
    """Every test starts with an empty occlusion cube-map cache, as a fresh shard process does. core/occlusion's
    CubeCache keyed a map by its origin rounded to 0.1 mm, so a map built earlier in this process for an origin less
    than 0.1 mm away could be reused: measured, run 1 of an 8-run Latin hypercube read 69.2431 % after the other tests
    of this file and 69.2448 % from a cold cache (and in every shard). The origin is keyed exactly now (integration of
    research tools part 2); the comparisons here are still of cold runs."""
    from dexvantage.core import occlusion as OC
    OC.CACHE.clear()
    yield


def _so100():
    from dexvantage.core.robot.samples import sample_project
    return sample_project('so100')


def _design(n=5, seed=2):
    return {'factors': [{'path': CAM + '.pitch', 'lo': -5, 'hi': 5}, {'path': CAM + '.z', 'lo': 0.3, 'hi': 0.45}],
            'responses': ['coverage.covered'], 'design': {'method': 'lhs', 'n': n, 'seed': seed},
            'analysis': {'bootstrap': 50}}


def _canon(doc, drop=HOW):
    return json.dumps({k: v for k, v in doc.items() if k not in drop}, sort_keys=True, allow_nan=True)


def _env(tmp_path):
    env = dict(os.environ)
    env['PYTHONPATH'] = ROOT + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env['QT_QPA_PLATFORM'] = 'offscreen'
    return env


def _cli(args, tmp_path, env=None, cwd=None):
    return subprocess.run([sys.executable, '-m', 'dexvantage', *args], capture_output=True, text=True, timeout=600,
                          env=env or _env(tmp_path), cwd=cwd)


# ============================================================================ partition
def test_shard_partition_covers_every_run_exactly_once_for_any_n():
    for m in range(0, 220):
        for n in range(1, 61):
            parts = C.partition(m, n)
            flat = [i for p in parts for i in p]
            assert len(flat) == m and sorted(flat) == list(range(m)), (m, n)
            assert len(parts) == n and max(map(len, parts)) - min(map(len, parts)) <= 1     # balanced
    assert C.shard_indices(10, 3, 4) == [3, 7]
    for bad in ((10, 4, 4), (10, -1, 4), (10, 0, 0)):
        with pytest.raises(C.ClusterError):
            C.shard_indices(*bad)


# ============================================================================ the shared disk cache
def test_disk_cache_round_trip_misses_and_refusals(tmp_path):
    c = C.DiskResultCache(str(tmp_path / 'cache'), size=2)
    key = ('coverage', 'a' * 64, 'b' * 64, 'c' * 64, 0, '')
    doc = {'x': 1.0000000000000002, 'n': 3, 'nan': float('nan'), 'l': [1, 'two', None, True], 'd': {'k': -0.0}}
    c.put(key, doc)
    assert c.disk_writes == 1
    fresh = C.DiskResultCache(str(tmp_path / 'cache'))
    got = fresh.get(key)
    assert CA.same(got, doc) and fresh.disk_hits == 1 and fresh.hits == 1 and fresh.misses == 0
    assert fresh.get(('coverage', 'other')) is None and fresh.misses == 1
    # a file cut short, corrupted, or holding another key reads as a miss, never as a result
    path = c.path(key)
    raw = open(path, 'rb').read()
    for broken in (raw[:-5], raw[:-5] + b'xxxxx', raw.replace(b'"n":3', b'"n":4')):
        with open(path, 'wb') as fh:
            fh.write(broken)
        assert CA.read_entry(path, key) is None and C.DiskResultCache(str(tmp_path / 'cache')).get(key) is None
    with open(path, 'wb') as fh:
        fh.write(raw)
    assert CA.read_entry(path, ('coverage', 'not this key')) is None
    # results that do not survive JSON exactly stay in memory only
    for inexact in ({'t': (1, 2)}, {1: 'int key'}, {'a': __import__('numpy').float32(1.5)}):
        k2 = ('s', repr(inexact))
        c.put(k2, inexact)
        assert not os.path.exists(c.path(k2))
    assert c.memory_only == 3
    # leftovers of killed writers are pruned
    stale = os.path.join(os.path.dirname(path), '.x.123.abc' + CA.TMP_SUFFIX)
    open(stale, 'w').close()
    os.utime(stale, (0, 0))
    assert CA.prune(str(tmp_path / 'cache'), older_than_s=60) == 1 and not os.path.exists(stale)


def test_run_study_through_the_disk_cache(tmp_path):
    p = _so100()
    c = C.DiskResultCache(str(tmp_path / 'cache'))
    a = jobs.run_study('coverage', p.snapshot(), cache=c)
    b = jobs.run_study('coverage', p.snapshot(), cache=C.DiskResultCache(str(tmp_path / 'cache')))
    assert not a['cached'] and b['cached'] and _canon(a) == _canon(b)


def test_concurrent_cache_writers_produce_no_torn_files(tmp_path):
    from dexvantage.validation.checks_cluster import concurrent_writers
    r = concurrent_writers(str(tmp_path / 'cache'), writers=4, n_keys=16, rounds=2)
    assert r['bad_reads'] == 0 and r['torn_files'] == 0 and r['temporary_files_left'] == 0
    assert r['exit_codes'] == [0, 0, 0, 0]


def test_atomic_write_never_shows_a_partial_file(tmp_path):
    path = str(tmp_path / 'f.bin')
    CA.atomic_write(path, b'a' * 100)
    CA.atomic_write(path, b'b' * 200)
    assert open(path, 'rb').read() == b'b' * 200
    assert os.listdir(tmp_path) == ['f.bin']


# ============================================================================ the Slurm script
def test_sbatch_script_passes_bash_n_and_its_directives_parse(tmp_path):
    bash = shutil.which('bash')
    cases = [({}, 1), ({}, 7), ({'cpus_per_task': 8, 'mem': '16g', 'time': '2-00:00', 'partition': 'gpu-a',
                                 'account': 'lab.x', 'gpus': 1, 'throttle': 5,
                                 'activate': 'source "my env/bin/activate"',
                                 'setup': ['module purge', 'module load python/3.12'], 'python': 'python3',
                                 'extra': ['--qos=short', '--exclusive']}, 300)]
    for given, n in cases:
        st = SL.read_settings(given)
        text = SL.sbatch_script(st, n, title='test', items=n)
        d = SL.directives(text)
        dd = dict(d)
        assert dd['--array'] == (f'0-{n - 1}' if n > 1 else '0') + (f'%{given["throttle"]}' if 'throttle' in given
                                                                     else '')
        assert dd['--cpus-per-task'] == str(given.get('cpus_per_task', 1))
        assert dd['--mem'] == given.get('mem', '4G').upper() and dd['--time'] == given.get('time', '02:00:00')
        assert dd['--output'] == 'logs/shard-%A_%a.out' and dd['--job-name'] == 'dexvantage'
        assert ('--partition' in dd) == ('partition' in given) and ('--gpus' in dd) == ('gpus' in given)
        if 'extra' in given:
            assert dd['--qos'] == 'short' and ('--exclusive', None) in d
        assert f'--of {n} --out out' in text and '"$SHARD"' in text
        assert '/home/' not in text and '/Users/' not in text
        if bash:
            f = tmp_path / f'run{n}.sbatch'
            f.write_text(text)
            r = subprocess.run([bash, '-n', str(f)], capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
    # every directive comes before the first command (Slurm stops reading them there)
    lines = text.splitlines()
    first_cmd = next(i for i, ln in enumerate(lines) if ln.strip() and not ln.startswith('#'))
    assert all(i < first_cmd for i, ln in enumerate(lines) if ln.startswith('#SBATCH'))
    assert SL.directives('#!/bin/bash\n#SBATCH --mem=1G\necho hi\n#SBATCH --mem=9G\n') == [('--mem', '1G')]


def test_slurm_settings_are_checked_in_sentences():
    for bad, word in (({'mem': '4 gigs'}, 'Slurm size'), ({'time': 'two hours'}, 'Slurm time'),
                      ({'cpus_per_task': 0}, 'at least 1'), ({'partition': 'a b'}, 'letters'),
                      ({'extra': ['qos=short']}, 'start with --'), ({'activate': 'a\nb'}, 'one line'),
                      ({'nodes': 2}, 'Unknown Slurm setting')):
        with pytest.raises(SL.SlurmError, match=word):
            SL.read_settings(bad)
    st = SL.read_settings({'mem': '8G'})
    assert 'mem' not in st['defaults'] and 'time' in st['defaults'] and st['activate'] == ''


def test_job_folder_layout_and_refusals(tmp_path):
    p = _so100()
    w = C.experiment_work(p, _design(n=4))
    with pytest.raises(SL.SlurmError, match='MaxArraySize'):
        SL.write_job_folder(w, str(tmp_path / 'big'), {'max_array': 2}, shards=4)
    job = SL.write_job_folder(w, str(tmp_path / 'job'), {'activate': 'source venv/bin/activate'}, shards=50)
    assert job['shards'] == 5                                   # never more shards than items
    names = set(os.listdir(tmp_path / 'job'))
    assert {'work.json', 'run.sbatch', 'job.json', 'README.txt', 'out', 'logs', 'cache', 'projects'} <= names
    assert os.stat(tmp_path / 'job' / 'run.sbatch').st_mode & stat.S_IXUSR
    with pytest.raises(SL.SlurmError, match='not empty'):
        SL.write_job_folder(w, str(tmp_path / 'job'), shards=2)
    jf = json.loads((tmp_path / 'job' / 'job.json').read_text())
    assert jf['shards'] == 5 and jf['work_id'] == w.work_id and 'mem' in jf['defaults']
    # a work file edited by hand, or a project changed, is refused
    wd = json.loads((tmp_path / 'job' / 'work.json').read_text())
    wd['items'][1]['values'][CAM + '.pitch'] = 99.0
    (tmp_path / 'job' / 'work.json').write_text(json.dumps(wd))
    with pytest.raises(C.ClusterError, match='edited'):
        C.Work.load(str(tmp_path / 'job'))
    job2 = SL.write_job_folder(w, str(tmp_path / 'job2'), shards=2)
    q = _so100()
    q.name = 'changed'
    q.save(str(tmp_path / 'job2' / 'projects' / 'base.dexv'))
    with pytest.raises(C.ClusterError, match='changed since'):
        C.Work.load(job2['work'])


# ============================================================================ gather against serial, separate processes
def test_gather_of_separate_process_shards_equals_the_serial_experiment(tmp_path):
    p = _so100()
    d = _design(n=6, seed=7)
    ser = EX.run(p, d)
    w = C.experiment_work(p, d)
    job = str(tmp_path / 'job')
    SL.write_job_folder(w, job, shards=3)
    env = _env(tmp_path)
    procs = [subprocess.Popen([sys.executable, '-m', 'dexvantage', '--run-shard', 'work.json', '--shard', str(i),
                               '--of', '3', '--out', 'out'], cwd=job, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True) for i in range(3)]
    for pr in procs:
        out, err = pr.communicate(timeout=600)
        assert pr.returncode == 0, err
        assert 'Traceback' not in err and 'done' in out
    r = _cli(['--gather', job], tmp_path)
    assert r.returncode == 0, r.stderr
    got = json.loads((tmp_path / 'job' / 'result.json').read_text())
    assert (tmp_path / 'job' / 'result.csv').exists() and 'provenance' in got
    assert _canon(got, HOW + ('provenance', 'files')) == _canon(ser, HOW)         # bit for bit, as JSON
    assert got['execution']['mode'].startswith('Slurm job array of 3 shards')
    assert got['execution']['runs_done'] == 6 and got['execution']['peak_rss_mb'] > 0
    # the in-memory gather gives the same document as the file (before provenance)
    rep = C.gather(job, write=False)
    assert rep['complete'] and _canon(rep['result']) == _canon(ser, HOW)


def test_gather_of_separate_process_shards_equals_the_serial_tolerance_study(tmp_path):
    p = _so100()
    p.analysis.studies['tolerance'] = {'n': 7, 'seed': 11, 'metrics': ['coverage.covered'],
                                       'distribution': 'normal'}
    ser = jobs.run_study('tolerance', p.snapshot())
    w = C.tolerance_work(p)
    job = str(tmp_path / 'job')
    SL.write_job_folder(w, job, shards=4)
    for i in range(4):
        r = _cli(['--run-shard', os.path.join(job, 'work.json'), '--shard', str(i), '--of', '4'], tmp_path)
        assert r.returncode == 0, r.stderr
    rep = C.gather(job)
    assert rep['complete']
    assert _canon(rep['result']) == _canon(ser)
    assert rep['result']['metrics']['coverage.covered']['values'] == ser['metrics']['coverage.covered']['values']
    assert rep['result']['draws'] == ser['draws']


# ============================================================================ missing, failed, resumed, foreign
def _bad_study_module(tmp_path):
    mod = tmp_path / 'cluster_toy_mod.py'
    mod.write_text('from dexvantage.core import jobs\n\n\n'
                   '@jobs.study("_cluster_toy", "Cluster toy", quick=False)\n'
                   'def toy(project, ctx):\n'
                   '    if project.name == "Bad":\n'
                   '        raise RuntimeError("boom")\n'
                   '    return {"name": project.name, "n": len(project.name)}\n')
    sys.path.insert(0, str(tmp_path))
    import importlib
    importlib.import_module('cluster_toy_mod')
    return mod


def test_missing_and_failed_shards_are_reported_with_a_resubmit_script(tmp_path):
    from dexvantage.core.project import blank_project
    _bad_study_module(tmp_path)
    try:
        projs = [blank_project(n) for n in ('A', 'Bad', 'C', 'D', 'E', 'F')]
        w = C.studies_work([(p, '_cluster_toy', {}) for p in projs])
        assert w.modules == ['cluster_toy_mod']                   # a shard process imports it
        job = str(tmp_path / 'job')
        SL.write_job_folder(w, job, shards=3)
        rep = C.gather(job)
        assert not rep['complete'] and rep['missing'] == [0, 1, 2] and 'resubmit.sbatch' in rep['text']
        s0 = C.run_shard(os.path.join(job, 'work.json'), 0, 3)
        assert s0['done'] == 2 and s0['failed'] == 0
        s1 = C.run_shard(os.path.join(job, 'work.json'), 1, 3)     # items 1 (Bad) and 4
        assert s1['done'] == 1 and s1['failed'] == 1
        rep = C.gather(job)
        assert not rep['complete'] and rep['failed'] == [1] and rep['missing'] == [2]
        row = next(r for r in rep['rows'] if r['shard'] == 1)
        assert 'Job 1 stopped with an unexpected error (RuntimeError: boom)' in row['error']
        rs = open(rep['resubmit_path']).read()
        assert dict(SL.directives(rs))['--array'] == '1-2' and '--of 3' in rs
        if shutil.which('bash'):
            assert subprocess.run(['bash', '-n', rep['resubmit_path']]).returncode == 0
        assert 'First failure' in rep['text']
        # a shard killed part way leaves a cut line: skipped, the item is simply not done
        f = os.path.join(job, 'out', 'shard-00002-of-00003.deadbeef0000.jsonl')
        SH.append_line(f, {'type': 'header', 'work_id': w.work_id, 'shard': 2, 'of': 3, 'items': 2})
        with open(f, 'a') as fh:
            fh.write('{"type": "item", "index": 2, "status": "ok", "result": {"doc": {"na')
        rep = C.gather(job)
        assert rep['incomplete'] == [2] and rep['items_done'] == 3
        # the resubmitted shard 2 skips nothing it has not done; shard 0 run again skips both its items
        assert C.run_shard(os.path.join(job, 'work.json'), 0, 3)['skipped'] == 2
        assert C.run_shard(os.path.join(job, 'work.json'), 2, 3)['done'] == 2
        # outputs of another work in the folder are ignored, and said
        other = C.studies_work([(blank_project('Z'), 'summary', {})])
        SH.append_line(os.path.join(job, 'out', 'shard-00000-of-00003.0123456789ab.jsonl'),
                       {'type': 'header', 'work_id': other.work_id, 'shard': 0, 'of': 3})
        st = C.status(job)
        assert st['foreign'] == ['shard-00000-of-00003.0123456789ab.jsonl']
        assert st['failed'] == [1] and st['items_done'] == 5
    finally:
        sys.path.remove(str(tmp_path))
        jobs.STUDIES.pop('_cluster_toy', None)
        sys.modules.pop('cluster_toy_mod', None)


def test_gather_cli_exit_codes_and_sentences(tmp_path):
    p = _so100()
    job = str(tmp_path / 'job')
    SL.write_job_folder(C.experiment_work(p, _design(n=3)), job, shards=2)
    r = _cli(['--gather', job], tmp_path)
    assert r.returncode == 1 and 'of 2 shards, 2 missing (0-1)' in r.stdout and 'Traceback' not in r.stderr
    for args, word in ((['--gather', str(tmp_path)], 'could not be read'),
                       (['--run-shard', os.path.join(job, 'work.json'), '--shard', '9', '--of', '2'], 'does not exist'),
                       (['--run-shard', os.path.join(job, 'work.json')], 'needs --shard'),
                       (['--shard', '1'], 'go with --run-shard'),
                       (['--sample', 'so100', '--jobs', '2', '--slurm', str(tmp_path / 'x'), '--experiment', 'd.json'],
                        'give one of'),
                       (['--sample', 'so100', '--jobs', '2'], 'go with one of'),
                       (['--sample', 'so100', '--slurm', str(tmp_path / 'y'), '--tolerance-mc', 'o.json',
                         '--slurm-settings', '{"mem": "lots"}'], 'Slurm size'),
                       (['--submit'], 'goes with --slurm')):
        r = _cli(args, tmp_path)
        assert r.returncode == 2, (args, r.stdout, r.stderr)
        assert word in r.stderr and 'Traceback' not in r.stderr, (args, r.stderr)


# ============================================================================ mock sbatch
def _mock_sbatch(tmp_path):
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    log = tmp_path / 'sbatch.log'
    sb = bindir / 'sbatch'
    sb.write_text('#!/bin/bash\n'
                  f'echo "$PWD|$*" >> "{log}"\n'
                  'n=$(wc -l < "' + str(log) + '")\n'
                  'echo "$((4100 + n));testcluster"\n')
    sb.chmod(0o755)
    env = _env(tmp_path)
    env['PATH'] = str(bindir) + os.pathsep + env.get('PATH', '')
    return env, log


def test_slurm_folder_and_submit_with_a_mock_sbatch(tmp_path):
    env, log = _mock_sbatch(tmp_path)
    design = tmp_path / 'design.json'
    design.write_text(json.dumps(_design(n=4)))
    job = str(tmp_path / 'job')
    r = _cli(['--sample', 'so100', '--experiment', str(design), '--slurm', job, '--shards', '2', '--submit',
              '--slurm-settings', json.dumps({'mem': '3G', 'partition': 'cpu'})], tmp_path, env=env)
    assert r.returncode == 0, r.stderr
    assert 'Submitted array job 4101.' in r.stdout
    cwd, args = log.read_text().strip().split('|')
    assert os.path.samefile(cwd, job) and args == '--parsable run.sbatch'
    d = dict(SL.directives(open(os.path.join(job, 'run.sbatch')).read()))
    assert d['--mem'] == '3G' and d['--partition'] == 'cpu' and d['--array'] == '0-1'
    # what an array task runs: the script itself with Slurm's variables (the python of this venv in its place)
    script = open(os.path.join(job, 'run.sbatch')).read().replace('exec python -m', f'exec {sys.executable} -m')
    for i in range(2):
        e = dict(env, SLURM_ARRAY_TASK_ID=str(i), SLURM_SUBMIT_DIR=job)
        rr = subprocess.run(['bash', '-c', script], env=e, capture_output=True, text=True, timeout=600, cwd=job)
        assert rr.returncode == 0, rr.stderr
    rep = C.gather(job)
    assert rep['complete'] and rep['result']['execution']['runs_planned'] == 4
    # without sbatch on the PATH: a sentence
    with pytest.raises(SL.SlurmError, match='sbatch is not on the PATH'):
        SL.submit(job, sbatch='no-such-sbatch-here')


def test_optimiser_rounds_on_slurm_equal_the_serial_search(tmp_path):
    env, log = _mock_sbatch(tmp_path)
    p = _so100()
    st = {'budget': 12, 'population': 4, 'objectives': ['coverage.covered']}
    ser = C.optimise(p, st, jobs=1)
    folder = str(tmp_path / 'opt')
    p.save(str(tmp_path / 'so100.dexv'))
    r = _cli([str(tmp_path / 'so100.dexv'), '--optimise', str(tmp_path / 'o.json'), '--slurm', folder, '--shards', '2',
              '--optimise-settings', json.dumps(st), '--submit'], tmp_path, env=env)
    assert r.returncode == 0, r.stderr
    assert 'Round 1:' in r.stdout and 'Submitted (array 4101, then the continue job 4102)' in r.stdout
    lines = log.read_text().strip().splitlines()
    assert lines[1].endswith('--parsable --dependency=afterany:4101 continue.sbatch')
    if shutil.which('bash'):
        assert subprocess.run(['bash', '-n', os.path.join(folder, 'continue.sbatch')]).returncode == 0
    with pytest.raises(C.ClusterError, match='not complete'):            # continuing before the round ran
        C.optimise_slurm(folder)
    for _k in range(10):
        rd = sorted(d for d in os.listdir(folder) if d.startswith('round-'))[-1]
        for i in range(2):
            C.run_shard(os.path.join(folder, rd), i, 2)
        res = C.optimise_slurm(folder)
        if res['status'] == 'done':
            break
    assert res['status'] == 'done' and res['round'] == 3
    got = json.loads(open(res['result_path']).read())
    assert _canon(got, ('search', 'schema', 'study', 'provenance')) == _canon(ser, ('search',))
    assert got['search']['cluster']['rounds'] == 3


# ============================================================================ this machine
def test_local_pool_equals_serial_and_states_its_memory():
    p = _so100()
    d = _design(n=6, seed=3)
    ser = EX.run(p, d)
    par = C.experiment(p, d, jobs=2)
    assert _canon(par) == _canon(ser, HOW)
    ex = par['execution']
    assert ex['mode'] == 'process pool of 2 (core/cluster)' and ex['workers'] == 2
    assert 50 < ex['peak_rss_mb_per_worker'] < ex['mem_per_worker_mb_planned']
    one = C.experiment(p, d, jobs=1)
    assert _canon(one) == _canon(ser, HOW) and one['execution']['mode'] == 'serial (core/cluster)'
    from dexvantage.core.cluster import local as L
    assert 1 <= L.recommended_workers() <= (os.cpu_count() or 1)
    assert L.recommended_workers(mem_per_worker_mb=1e12) == 1
    assert 'GiB each (a planning default' in L.memory_statement(3)


def test_local_pool_cancel_keeps_results_and_resumes(tmp_path):
    from dexvantage.core.cluster.local import run_items
    p = _so100()
    w = C.experiment_work(p, _design(n=8, seed=1))
    seen = []
    ctx = jobs.Ctx(progress=lambda f, t='': seen.append(f), cancel=lambda: len(seen) >= 2)
    out = str(tmp_path / 'out')
    r = run_items(w, 2, ctx, out_dir=out)
    assert r['cancelled'] and 2 <= len(r['results']) < len(w.items)
    with pytest.raises(C.ClusterError, match='Cancelled after'):
        C.experiment(p, _design(n=8, seed=1), jobs=2, ctx=jobs.Ctx(cancel=lambda: True))
    done = len(SH.read_outputs(out, w)['results'])
    assert done >= len(r['results'])
    r2 = run_items(w, 2, out_dir=out)                 # resumes: only the items not kept are run
    assert not r2['cancelled'] and len(r2['results']) == len(w.items)
    from dexvantage.validation.checks_cluster import leaf_diffs
    got, want = C.assemble(w, r2['results']), EX.run(p, _design(n=8, seed=1))
    assert leaf_diffs(json.loads(_canon(got)), json.loads(_canon(want))) == []


def test_optimiser_generations_in_a_pool_equal_the_serial_search():
    from dexvantage.core.optimise import rig
    p = _so100()
    st = {'budget': 12, 'population': 4, 'objectives': ['coverage.covered']}
    q = p.snapshot()
    q.analysis.studies['optimise'] = {**(q.analysis.studies.get('optimise') or {}), **st}
    plain = rig.optimise(q, rig.settings_of(q))
    par = C.optimise(p, st, jobs=2)
    assert _canon(par, ('search',)) == _canon(plain, ('search',))
    assert par['search']['evaluations'] == plain['search']['evaluations'] == 12
    assert par['search']['cluster']['designs_in_pool'] >= 6 and par['search']['studies_scored'] < \
        plain['search']['studies_scored']


def test_cluster_studies_cli_and_the_jobs_list(tmp_path):
    from dexvantage.core.project import blank_project
    for n in ('a', 'b'):
        blank_project(f'Cell {n}').save(str(tmp_path / f'{n}.dexv'))
    lst = tmp_path / 'jobs.json'
    lst.write_text(json.dumps([{'project': 'a.dexv', 'study': 'summary'}, {'project': 'b.dexv', 'study': 'summary'},
                               {'project': 'a.dexv', 'study': 'bandwidth', 'settings': {}}]))
    r = _cli(['--cluster-studies', str(lst), '--jobs', '2'], tmp_path)
    assert r.returncode == 0, r.stderr
    doc = json.loads((tmp_path / 'jobs.results.json').read_text())
    assert [j['study'] for j in doc['jobs']] == ['summary', 'summary', 'bandwidth']
    assert doc['jobs'][1]['doc']['project'] == 'Cell b' and 'provenance' in doc
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps([{'project': 'a.dexv', 'study': 'nope'}]))
    r = _cli(['--cluster-studies', str(bad), '--jobs', '1'], tmp_path)
    assert r.returncode == 2 and 'no study called "nope"' in r.stderr


def test_jobs_flag_runs_the_experiment_and_tolerance_batch(tmp_path):
    design = tmp_path / 'design.json'
    design.write_text(json.dumps(_design(n=3)))
    r = _cli(['--sample', 'so100', '--experiment', str(design), '--jobs', '2', '--experiment-out',
              str(tmp_path / 'e.json')], tmp_path)
    assert r.returncode == 0, r.stderr
    e = json.loads((tmp_path / 'e.json').read_text())
    assert e['execution']['workers'] == 2 and (tmp_path / 'e.csv').exists() and 'provenance' in e
    ser = EX.run(_so100(), _design(n=3))
    assert _canon(e, HOW + ('provenance', 'files')) == _canon(ser, HOW)


# ============================================================================ API, assistant
def test_dv_cluster_api(tmp_path):
    import dexvantage as dv
    rig = dv.sample('so100')
    res = dv.cluster.experiment(rig, _design(n=3), jobs=1)
    assert isinstance(res, dv.Result) and res.study == 'experiments' and res['execution']['runs_done'] == 3
    tol = dv.cluster.tolerance(rig, jobs=1, n=3, metrics=['coverage.covered'])
    assert tol.study == 'tolerance' and tol['done'] == 3
    out = dv.cluster.studies([(rig, 'summary', {}), (rig, 'bandwidth', None)], jobs=1)
    assert [r.study for r in out] == ['summary', 'bandwidth']
    job = dv.cluster.slurm(rig, tmp_path / 'job', experiment=_design(n=3), shards=2, mem='2G')
    assert job['shards'] == 2 and os.path.exists(job['script'])
    st = dv.cluster.status(tmp_path / 'job')
    assert st['missing'] == [0, 1]
    for i in range(2):
        dv.cluster.run_shard(job['work'], i, 2)
    rep = dv.cluster.gather(tmp_path / 'job')
    assert rep['complete'] and rep['result']['execution']['runs_done'] == 3
    assert dv.cluster.partition(5, 2) == [[0, 2, 4], [1, 3]]
    assert {r['key'] for r in dv.cluster.settings()} >= {'mem', 'time', 'activate'}
    assert 'dv.cluster' in repr(dv.cluster)
    with pytest.raises(dv.DexVantageError, match='one of experiment='):
        dv.cluster.slurm(rig, tmp_path / 'j2')
    # no Qt on this path
    code = ('import sys, dexvantage as dv; dv.cluster.partition(3, 2); import dexvantage.core.cluster.local; '
            'print("PyQt6" in sys.modules)')
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=_env(tmp_path))
    assert r.stdout.strip() == 'False', r.stderr


def test_assistant_reads_a_cluster_job(tmp_path):
    from dexvantage.assistant import tools as T
    from dexvantage.assistant.session import FileSession
    p = _so100()
    job = str(tmp_path / 'job')
    SL.write_job_folder(C.experiment_work(p, _design(n=3)), job, shards=2)
    s = FileSession(project=p)
    out = T.call(s, 'get_cluster_job', {'folder': job})
    assert out['missing'] == [0, 1] and not out['complete'] and out['items'] == 4
    for i in range(2):
        C.run_shard(os.path.join(job, 'work.json'), i, 2)
    C.gather(job)
    out = T.call(s, 'get_cluster_job', {'folder': job})
    assert out['complete'] and out['headline']['word']
    assert 'error' in T.call(s, 'get_cluster_job', {'folder': str(tmp_path / 'nothing')})
