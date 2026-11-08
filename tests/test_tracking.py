"""The experiment tracker (core/tracking): records, the run store (round trip, filters, updates, several writers,
crashes), the provenance hash, the metric extraction per run kind, comparisons with the core/stats hook, re-runs,
CSV / Parquet / MLflow / W&B exports, batch --track / --runs / --runs-export, the Python API and the assistant's
reading tool."""
import json
import os
import struct
import subprocess
import sys
import textwrap

import pytest

from dexvantage.core import jobs, provenance as PV, tracking as TR
from dexvantage.core.project import blank_project
from dexvantage.core.tracking import mlruns as ML, record as RC, tables as TB, wandb_replay as WB
from dexvantage.validation import checks_tracking as CT

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


class _Ctx:
    """The bits of a validation Ctx the tracker's helpers use."""

    def __init__(self, tmp):
        import numpy as np
        self._tmp = str(tmp)
        self.cache = {}
        self.quick = True
        self._np = np

    def rng(self, cid, k=0):
        import zlib
        return self._np.random.default_rng([7, zlib.crc32(cid.encode()), k])

    def tmp(self):
        return self._tmp


@pytest.fixture
def saved(tmp_path):
    """A blank project saved to tmp_path/cell.dexv (so its run store goes beside it)."""
    p = blank_project('Tracked cell')
    path = str(tmp_path / 'cell.dexv')
    p.save(path)
    from dexvantage.core.project import Project
    return Project.load(path)


# ====================================================================== records and the store
def test_record_fields_and_store_round_trip(tmp_path, saved):
    doc = jobs.run_study('rig', saved.snapshot())
    st = TR.RunStore(str(tmp_path / 'store'))
    rec = TR.log('rig', doc, saved, store=st, interface='api', tags=['a', 'b, c', 'a'], note='two\nlines')
    for k in ('run_id', 'kind', 'study', 'started_utc', 'ended_utc', 'status', 'project', 'variant', 'settings',
              'seeds', 'metrics', 'headline', 'result', 'provenance', 'tags', 'note'):
        assert k in rec
    assert rec['schema'] == RC.SCHEMA and len(rec['run_id']) == 32 and rec['kind'] == 'study'
    assert rec['tags'] == ['a', 'b c'] and rec['note'] == 'two\nlines' and rec['status'] == 'finished'
    assert rec['project']['sha256'] == PV.project_digest(saved)['sha256']
    assert rec['project']['file'] == 'cell.dexv' and rec['variant']['name'] == saved.variant.name
    assert rec['provenance']['dexvantage']['version'] and rec['provenance']['runtime']['numpy']
    m = rec['metrics']['rig.camera_count']
    assert m['value'] == 0.0 and m['unit'] == 'count' and m['status'] == 'ok'
    assert rec['metrics']['rig.cost']['value'] is None and rec['metrics']['rig.cost']['status'] == 'not stated'
    back = TR.RunStore(st.folder)
    assert back.get(rec['run_id']) == json.loads(json.dumps(rec))
    res = back.result(rec['run_id'])
    assert PV.verify(res)['ok'] is True and res['study'] == 'rig'
    # no path, user name or host name: the store folder and the project's folder appear nowhere in what is stored
    for root, _d, names in os.walk(st.folder):
        for n in names:
            text = open(os.path.join(root, n), encoding='utf-8').read()
            assert str(tmp_path) not in text, n


def test_validation_checks_pass(tmp_path):
    c = _Ctx(tmp_path)
    for fn in (CT.store_round_trip, CT.crash_recovery, CT.csv_parquet_identical, CT.mlflow_reader):
        out = fn(c)
        assert out.error == 0, (fn.__name__, out.detail)
    assert CT.store_round_trip(_Ctx(tmp_path / 'b')).detail['fields_compared'] > 300


def test_concurrent_writer_processes_lose_nothing(tmp_path):
    r = CT.concurrent(str(tmp_path / 'store'), 6, 40)
    assert r['errors'] == [] and r['reported'] == 240 and r['stored'] == 240
    assert r['lost'] == r['extra'] == r['corrupt'] == r['damaged_lines'] == r['temporary_left'] == 0
    assert r['writers_at_once'] >= 2                                   # the writes really overlapped


def test_filters_updates_and_delete(tmp_path):
    from dexvantage.core.tracking._stress import synthetic
    st = TR.RunStore(str(tmp_path / 's'))
    ids = []
    for i, (day, study) in enumerate([('2026-11-01', 'a'), ('2026-11-03', 'b'), ('2026-11-05', 'a')]):
        rec, res = synthetic(0, i)
        rec.update(study=study, started_utc=f'{day}T12:00:00.000Z', tags=[f't{i}', 'all'])
        ids.append(st.add(rec, res)['run_id'])
    assert [r['run_id'] for r in st.runs(study='a')] == [ids[0], ids[2]]
    assert [r['run_id'] for r in st.runs(tag='t1')] == [ids[1]]
    assert [r['run_id'] for r in st.runs(since='2026-11-02', until='2026-11-03')] == [ids[1]]   # until: whole day
    assert len(st.runs(tag=['t0', 't2'])) == 2
    with pytest.raises(TR.TrackingError, match='not a date'):
        st.runs(since='last tuesday')
    st.update(ids[1], add_tags=['new'], note='checked')
    assert st.get(ids[1])['tags'] == ['t1', 'all', 'new'] and st.runs(tag='new')[0]['note'] == 'checked'
    st.update(ids[1], remove_tags=['all'])
    assert st.get(ids[1])['tags'] == ['t1', 'new']
    st.delete(ids[0])
    assert [r['run_id'] for r in st.runs()] == ids[1:]
    assert os.path.isdir(os.path.join(st.folder, 'deleted', ids[0]))
    with pytest.raises(TR.TrackingError, match='no run'):
        st.get(ids[0])
    with pytest.raises(TR.TrackingError, match='not a run id'):
        st.get('../../etc')


def test_store_for_and_settings(tmp_path, saved):
    assert TR.settings_of(saved) == TR.DEFAULTS and not TR.enabled(saved)
    assert TR.store_folder(saved) == str(tmp_path / 'cell.runs')
    saved.analysis.studies['tracking'] = {'enabled': True, 'folder': 'elsewhere', 'tags': 'x, y'}
    assert TR.store_folder(saved) == str(tmp_path / 'elsewhere') and TR.settings_of(saved)['tags'] == ['x', 'y']
    with pytest.raises(TR.TrackingError, match='not saved'):
        TR.store_for(blank_project('Unsaved'))
    p = blank_project('Unsaved')
    p.analysis.studies['tracking'] = {'enabled': True}
    assert not TR.should_log('tracking', p) and not TR.should_log('summary', p) and TR.should_log('rig', p)
    assert not TR.should_log('bandwidth', p, requested=False)              # a quick study re-run after an edit
    p.analysis.studies['tracking']['quick'] = True
    assert TR.should_log('bandwidth', p, requested=False)


def test_tracking_setting_is_not_in_the_content_hash():
    p = blank_project('Hash')
    h0 = PV.project_digest(p)['sha256']
    p.analysis.studies['tracking'] = {'enabled': True, 'folder': 'x'}
    assert PV.project_digest(p)['sha256'] == h0
    p.analysis.studies['coverage'] = {'voxel_mm': 3}
    assert PV.project_digest(p)['sha256'] != h0                            # other study settings still count
    q = blank_project('Hash')
    q.analysis.studies['coverage'] = {'voxel_mm': 3}
    assert PV.project_digest(q)['sha256'] == PV.project_digest(p)['sha256']


# ====================================================================== figures per run kind
def test_metrics_of_each_run_kind():
    tol = {'done': 4, 'metrics': {'coverage.covered': {'title': 'Seen', 'unit': '%', 'better': 'higher',
                                                       'nominal': 90.0, 'nominal_status': 'ok',
                                                       'values': [89.0, None, 91.0, 92.0], 'p5': 89.2, 'p50': 91.0,
                                                       'p95': 91.9, 'mean': 90.67, 'std': 1.5}}}
    m, s = RC.metrics_of('tolerance', tol)
    assert m['coverage.covered']['value'] == 90.0 and m['coverage.covered@p5']['value'] == 89.2
    # samples by draw: the draw without a value stays as None, so two runs with one seed pair draw by draw
    assert m['coverage.covered@p95']['stat'] == 'p95' and s == {'coverage.covered': [89.0, None, 91.0, 92.0]}
    ex = {'responses': [{'id': 'coverage.covered', 'title': 'Seen', 'unit': '%', 'better': 'higher'}],
          'baseline': {'coverage.covered': {'value': 88.0, 'status': 'ok'}},
          'best': {'run': 3, 'response': 'coverage.covered', 'value': 93.5, 'unit': '%', 'better': 'higher'}}
    m, _ = RC.metrics_of('experiments', ex)
    assert m['coverage.covered']['value'] == 88.0 and m['coverage.covered@best']['value'] == 93.5
    op = {'objectives': [{'id': 'coverage.covered', 'title': 'Seen', 'unit': '%', 'better': 'higher'},
                         {'id': 'rig.camera_count', 'title': 'Cameras', 'unit': 'count', 'better': 'lower'}],
          'current': {'figures': {'coverage.covered': 80.0, 'rig.camera_count': 2}},
          'options': [{'label': 'A', 'figures': {'coverage.covered': 85.0, 'rig.camera_count': 3}},
                      {'label': 'B', 'figures': {'coverage.covered': 82.0, 'rig.camera_count': 1}}]}
    m, _ = RC.metrics_of('optimise', op)
    assert m['coverage.covered@best']['value'] == 85.0 and m['rig.camera_count@best']['value'] == 1.0
    assert m['optimise.options']['value'] == 2.0
    card = {'counts': {'pass': 1, 'fail': 1, 'not_evaluated': 0},
            'items': [{'uid': 'u1', 'metric': 'sync.skew_ms', 'value': 1.0, 'bound': 'at_least', 'status': 'pass',
                       'op': '<', 'threshold': 2},
                      {'uid': 'u2', 'metric': 'sync.skew_ms', 'value': 1.0, 'status': 'fail', 'op': '<',
                       'threshold': 0.5}]}
    m, _ = RC.metrics_of('requirements', card)
    assert m['sync.skew_ms']['check'] == 'pass' and m['sync.skew_ms']['bound'] == 'at_least'
    assert m['sync.skew_ms@u2']['check'] == 'fail' and m['requirements.fail']['value'] == 1.0
    assert RC.kind_of('requirements') == 'check' and RC.kind_of('design_agent') == 'optimise'
    m, _ = RC.metrics_of('coverage', {'unexpected': 'shape'})          # never a guess
    assert all(r['value'] is None and r['status'] == 'not stated' for r in m.values())


def test_failed_run_has_reason_and_no_result(tmp_path, saved):
    st = TR.RunStore(str(tmp_path / 's'))
    rec = TR.log_failure('coverage', 'Coverage needs a camera: add one.', saved, store=st)
    assert rec['status'] == 'failed' and rec['reason'].startswith('Coverage needs') and rec['result'] is None
    assert st.result(rec['run_id']) is None and rec['metrics'] == {}


# ====================================================================== compare, significance, re-run
def _tol_record(st, project, values, seed):
    doc = {'schema': 'dexvantage.tolerance/1', 'study': 'tolerance', 'done': len(values), 'n': len(values),
           'seed': seed, 'metrics': {'coverage.covered': {'title': 'Seen', 'unit': '%', 'better': 'higher',
                                                          'nominal': 90.0, 'nominal_status': 'ok', 'values': values,
                                                          'p5': min(values), 'p50': sorted(values)[len(values) // 2],
                                                          'p95': max(values), 'mean': sum(values) / len(values),
                                                          'std': 1.0}}}
    project.analysis.studies['tolerance'] = {'seed': seed}
    return st.get(TR.log('tolerance', doc, project, store=st)['run_id'])


def test_compare_deltas_differences_and_significance(tmp_path, saved):
    st = TR.RunStore(str(tmp_path / 's'))
    a = _tol_record(st, saved, [90.0, 91.0, 89.5, 90.5, 90.2, 89.8], 1)
    b = _tol_record(st, saved, [93.0, 94.0, 92.5, 93.5, 93.2, 92.8], 1)
    cmp = TR.compare([a, b])
    row = next(r for r in cmp['rows'] if r['key'] == 'coverage.covered@mean')
    assert row['deltas'][0] == 0.0 and row['deltas'][1] == pytest.approx(3.0) and row['best'] == 1
    assert row['relative'][1] == pytest.approx(3.0 / (541.0 / 6))
    base = next(r for r in cmp['rows'] if r['key'] == 'coverage.covered')
    assert base['same'] and base['best'] is None
    sig = base['significance']
    try:
        from dexvantage.core import stats  # noqa: F401
        assert sig['available'] and sig.get('p') is not None and sig.get('paired') is True
    except ImportError:
        assert not sig['available'] and 'core/stats' in sig['text']
    assert next(r for r in cmp['rows'] if r['key'] == 'coverage.covered@p5')['significance'] is None
    c = st.get(TR.log('rig', jobs.run_study('rig', saved.snapshot()), saved, store=st)['run_id'])
    cmp2 = TR.compare([a, c])
    assert any('different studies' in n for n in cmp2['notes'])
    assert any(d['path'].startswith('settings') for d in cmp2['differences'])
    one = TR.significance([1.0], [2.0], 'higher')
    assert not one['available'] and 'replicates' in one['text']
    with pytest.raises(TR.TrackingError, match='two or more'):
        TR.compare([a])


def test_rerun_reproduces_and_logs_a_new_run(tmp_path, saved):
    st = TR.store_for(saved)
    rec = TR.log('rig', jobs.run_study('rig', saved.snapshot()), saved, store=st)
    new, rep = TR.rerun(st, rec['run_id'], saved)
    assert rep['verdict'] == 'reproduces', rep.get('message')
    assert new['rerun_of'] == rec['run_id'] and new['reproduce']['verdict'] == 'reproduces' and 'rerun' in new['tags']
    assert len(st.runs()) == 2
    failed = TR.log_failure('coverage', 'No camera.', saved, store=st)
    with pytest.raises(TR.TrackingError, match='kept no result'):
        TR.rerun(st, failed['run_id'], saved)


# ====================================================================== exports
def _store_with_runs(tmp_path, saved, n=3):
    st = TR.RunStore(str(tmp_path / 'store'))
    for i in range(n):
        TR.log('rig' if i % 2 == 0 else 'bandwidth', jobs.run_study('rig' if i % 2 == 0 else 'bandwidth',
                                                                    saved.snapshot()),
               saved, store=st, tags=[f'n{i}'], note=f'run {i}')
    return st


def test_csv_and_parquet_hold_identical_values(tmp_path):
    import numpy as np
    runs = CT.table_records(np.random.default_rng(3), 25, 9)
    for form in ('wide', 'long'):
        TB.write_csv(runs, str(tmp_path / f'{form}.csv'), form)
        TB.write_parquet(runs, str(tmp_path / f'{form}.parquet'), form)
        cols, crow = TB.read_csv(str(tmp_path / f'{form}.csv'), form)
        pcols, prow = TB.read_parquet(str(tmp_path / f'{form}.parquet'))
        assert cols == pcols and len(crow) == len(prow)
        fl = TB.float_columns(cols, form)
        for a, b in zip(crow, prow):
            for c in cols:
                if c in fl:
                    pa = None if a[c] is None else struct.pack('<d', a[c])
                    pb = None if b[c] is None else struct.pack('<d', b[c])
                    assert pa == pb, c
                else:
                    assert (a[c] or None) == (b[c] or None), c
    import pyarrow.parquet as pq
    meta = pq.read_schema(str(tmp_path / 'wide.parquet')).field('m00.figure').metadata
    assert meta[b'unit'] == b'%'


def test_export_writes_every_format_with_a_manifest(tmp_path, saved):
    st = _store_with_runs(tmp_path, saved)
    out = TR.export(st, str(tmp_path / 'exp'))
    assert all(r['status'] == 'ok' for r in out['formats'].values()) and out['runs'] == 3
    man = json.loads((tmp_path / 'exp' / 'export_runs_manifest.json').read_text())
    assert man['schema'] == TR.EXPORT_SCHEMA and PV.verify(man)['ok'] is True
    assert {f['path'] for f in man['files']} >= {'runs.csv', 'runs.parquet', 'wandb/replay_wandb.py'}
    lay = ML.read_layout(str(tmp_path / 'exp' / 'mlruns'))
    assert set(lay['runs']) == {r['run_id'] for r in st.runs()} and '0' in lay['experiments']
    one = lay['runs'][st.runs()[0]['run_id']]
    assert one['meta']['status'] == 3 and one['meta']['source_type'] == 4 and one['tags']['mlflow.note.content']
    assert all(len(v[0]) == 3 for v in one['metrics'].values())
    with pytest.raises(TR.TrackingError, match='Unknown runs export format'):
        TR.export(st, str(tmp_path / 'x'), formats='csv,xls')
    with pytest.raises(TR.TrackingError, match='no run to export'):
        TR.export(st, str(tmp_path / 'y'), study='nothing')
    assert len(TR.export(st, str(tmp_path / 'z'), formats='csv', study='rig')['files']) == 3


def test_mlflow_names_are_valid():
    import re
    for k in ('coverage.covered@p5', 'a/b', '..x', 'tag.café wrist', 'x:y', 'sp ace', '.'):
        n = ML.mlflow_name(k)
        assert re.match(r'^[\w.\- :]*$', n) and '/' not in n and not n.startswith('.') and '..' not in n, (k, n)
    r = {'metrics': {'a@b': {}, 'a_b': {}}}
    assert len(set(ML.metric_names(r).values())) == 2


def test_mlflow_client_reads_the_store(tmp_path, saved):
    pytest.importorskip('mlflow', reason='mlflow-skinny (the tracking extra) is not installed')
    st = _store_with_runs(tmp_path, saved)
    TR.export(st, str(tmp_path / 'exp'), formats='mlflow')
    r = CT.mlflow_compare(str(tmp_path / 'exp' / 'mlruns'), st.runs(), reader='mlflow')
    assert r['reader'].startswith('MLflow') and r['differences'] == [] and r['read'] == 3, r


def test_mlflow_layout_without_mlflow(tmp_path, saved):
    st = _store_with_runs(tmp_path, saved)
    TR.export(st, str(tmp_path / 'exp'), formats='mlflow')
    r = CT.mlflow_compare(str(tmp_path / 'exp' / 'mlruns'), st.runs(), reader='layout')
    assert r['differences'] == [] and r['read'] == 3


FAKE_WANDB = textwrap.dedent('''
    import json, os
    LOG = os.environ['FAKE_WANDB_LOG']
    def _w(ev):
        with open(LOG, 'a') as fh:
            fh.write(json.dumps(ev) + "\\n")
    class _Summary(dict):
        def update(self, d):
            super().update(d); _w({'op': 'summary', 'd': dict(d)})
    class _Run:
        def __init__(self, kw):
            self.kw = kw; self.summary = _Summary()
        def log(self, d):
            _w({'op': 'log', 'd': d})
        def finish(self):
            _w({'op': 'finish'})
    def init(**kw):
        _w({'op': 'init', 'kw': kw, 'mode_env': os.environ.get('WANDB_MODE')}); return _Run(kw)
''')


def test_wandb_replay_calls_the_official_api(tmp_path, saved):
    st = _store_with_runs(tmp_path, saved, 2)
    out = TR.export(st, str(tmp_path / 'exp'), formats='wandb')
    assert out['formats']['wandb']['status'] == 'ok'
    data = WB.load(str(tmp_path / 'exp' / 'wandb' / 'wandb_runs.json'))
    assert data['schema'] == WB.SCHEMA and data['runs'][1]['config']['dexvantage.not_stated']   # bandwidth: no hosts
    fake = tmp_path / 'fake'
    (fake / 'wandb').mkdir(parents=True)
    (fake / 'wandb' / '__init__.py').write_text(FAKE_WANDB)
    log = tmp_path / 'calls.jsonl'
    env = {k: v for k, v in os.environ.items() if not k.startswith('WANDB')}
    env.update(PYTHONPATH=str(fake), FAKE_WANDB_LOG=str(log), HOME=str(tmp_path / 'home'))
    r = subprocess.run([sys.executable, str(tmp_path / 'exp' / 'wandb' / 'replay_wandb.py'), '--project', 'p1',
                        '--dir', str(tmp_path / 'wb')], capture_output=True, text=True, env=env, timeout=60,
                       cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    calls = [json.loads(x) for x in log.read_text().splitlines()]
    inits = [c for c in calls if c['op'] == 'init']
    runs = st.runs()
    assert [c['kw']['id'] for c in inits] == [x['run_id'] for x in runs]
    k = inits[0]['kw']
    assert k['project'] == 'p1' and k['mode'] == 'offline' and inits[0]['mode_env'] == 'offline'
    assert k['name'] == runs[0]['name'] and k['notes'] == 'run 0' and k['tags'] == ['n0'] and k['group'] == 'rig'
    assert k['job_type'] == 'study' and k['reinit'] == 'finish_previous' and k['config']['dexvantage.study'] == 'rig'
    logs = [c['d'] for c in calls if c['op'] == 'log']
    assert logs[0] == {m: v['value'] for m, v in runs[0]['metrics'].items() if v['value'] is not None}
    assert sum(c['op'] == 'finish' for c in calls) == 2
    # without wandb: one sentence, exit 2
    env.pop('PYTHONPATH')
    r = subprocess.run([sys.executable, '-I', str(tmp_path / 'exp' / 'wandb' / 'replay_wandb.py')],
                       capture_output=True, text=True, env=env, timeout=60, cwd=str(tmp_path))
    if r.returncode == 2:
        assert 'wandb package is not installed' in r.stderr and 'Traceback' not in r.stderr


def test_wandb_replay_offline_with_real_wandb(tmp_path, saved):
    pytest.importorskip('wandb', reason='wandb (the tracking extra) is not installed')
    st = _store_with_runs(tmp_path, saved, 2)
    TR.export(st, str(tmp_path / 'exp'), formats='wandb')
    h = tmp_path / 'home'
    env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'HOME': str(h), 'WANDB_MODE': 'offline',
           'WANDB_SILENT': 'true', 'WANDB_CONFIG_DIR': str(h / 'c'), 'WANDB_CACHE_DIR': str(h / 'k'),
           'WANDB_DATA_DIR': str(h / 'd'), 'WANDB_DISABLE_GIT': 'true', 'XDG_CONFIG_HOME': str(h / 'x')}
    r = subprocess.run([sys.executable, str(tmp_path / 'exp' / 'wandb' / 'replay_wandb.py'), '--dir',
                        str(tmp_path / 'wb')], capture_output=True, text=True, env=env, timeout=120,
                       cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr[-2000:]
    names = os.listdir(tmp_path / 'wb' / 'wandb')
    for run in st.runs():
        assert any(n.startswith('offline-run-') and n.endswith(run['run_id']) for n in names), names


# ====================================================================== the study, the metric, batch, API, tool
def test_tracking_study_summary_and_metric(tmp_path, saved):
    from dexvantage.core import metrics as M
    doc = jobs.run_study('tracking', saved.snapshot())
    assert doc['headline']['word'] == 'tracking off' and doc['suggestion']['action'] == 'enable'
    saved.analysis.studies['tracking'] = {'enabled': True}
    doc = jobs.run_study('tracking', saved.snapshot())
    assert doc['headline']['word'] == 'no runs yet' and doc['store'] == 'cell.runs'
    TR.log('rig', jobs.run_study('rig', saved.snapshot()), saved)
    doc = jobs.run_study('tracking', saved.snapshot())
    assert doc['headline']['word'] == 'tracking on' and doc['counts']['runs'] == 1 and doc['runs'][0]['study'] == 'rig'
    assert M.read(M.get('tracking.runs'), doc)['value'] == 1.0
    assert str(tmp_path) not in json.dumps(doc)
    p = blank_project('Unsaved')
    p.analysis.studies['tracking'] = {'enabled': True}
    doc = jobs.run_study('tracking', p)
    assert doc['headline']['word'] == 'not saved' and doc['headline']['status'] == 'serious'


def _batch(*args, cwd):
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    return subprocess.run([sys.executable, '-m', 'dexvantage', *args], capture_output=True, text=True, env=env,
                          cwd=cwd, timeout=180)


def test_batch_track_runs_and_export(tmp_path, saved):
    cwd = str(tmp_path)
    r = _batch('cell.dexv', '--rig-inventory', 'rig.json', '--track', 'runs', '--track-tag', 'ci', '--track-note',
               'nightly', cwd=cwd)
    assert r.returncode == 0, r.stderr
    assert 'logged run' in r.stdout
    r = _batch('cell.dexv', '--check', 'defaults', 'card.json', '--track', 'runs', cwd=cwd)
    assert r.returncode in (0, 1), r.stderr
    r = _batch('cell.dexv', '--grasp', 'g.json', '--track', 'runs', cwd=cwd)          # refused: logged as failed
    assert r.returncode == 2 and 'Traceback' not in r.stderr
    st = TR.RunStore(str(tmp_path / 'runs'))
    rows = st.runs()
    assert [x['kind'] for x in rows] == ['study', 'check', 'study']
    assert rows[0]['tags'] == ['ci'] and rows[0]['note'] == 'nightly' and rows[0]['interface'] == 'batch'
    assert rows[1]['study'] == 'requirements' and rows[2]['status'] == 'failed' and rows[2]['reason']
    assert rows[0]['result']['output']['option'] == '--rig-inventory'
    r = _batch('cell.dexv', '--bandwidth', '-', '--track', 'runs', cwd=cwd)
    assert r.returncode == 0 and json.loads(r.stdout)['study'] == 'bandwidth' and 'logged run' in r.stderr
    r = _batch('cell.dexv', '--runs', 'summary.json', '--track', 'runs', cwd=cwd)
    assert r.returncode == 0, r.stderr
    s = json.loads((tmp_path / 'summary.json').read_text())
    assert s['counts']['runs'] == 4 and s['headline']['word'] == 'tracking on'
    r = _batch('cell.dexv', '--runs-export', 'exp', '--track', 'runs', '--runs-formats', 'csv,mlflow', cwd=cwd)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / 'exp' / 'runs.csv').is_file() and (tmp_path / 'exp' / 'mlruns').is_dir()
    assert not (tmp_path / 'exp' / 'runs.parquet').exists()
    assert _batch('cell.dexv', '--track-note', 'x', cwd=cwd).returncode == 2
    assert _batch('cell.dexv', '--runs-export', 'e2', cwd=cwd).returncode == 2          # no store beside it
    r = _batch('cell.dexv', '--runs-export', 'e3', '--track', 'runs', '--runs-formats', 'xls', cwd=cwd)
    assert r.returncode == 2 and 'Unknown runs export format' in r.stderr


def test_api_runs_track_and_autolog(tmp_path):
    import dexvantage as dv
    rig = dv.sample('so100')
    rig.save(str(tmp_path / 'so100.dexv'))
    rig = dv.open(str(tmp_path / 'so100.dexv'))
    with pytest.raises(dv.DexVantageError, match='no run store'):
        dv.runs(rig)
    res = rig.run('rig')
    rec = dv.track(res, rig, tags=['manual'])
    assert rec['interface'] == 'api' and rec['tags'] == ['manual']
    rig.factors['setting:tracking.enabled'] = True
    rig.run('bandwidth')                                         # logged: the project's tracking is on
    df = dv.runs(rig)
    assert len(df) == 2 and 'rig.camera_count' in df.columns and df.attrs['units']['rig.camera_count'] == 'count'
    assert list(df['study']) == ['rig', 'bandwidth']
    assert len(dv.runs(str(tmp_path / 'so100.dexv'), study='rig')) == 1
    assert len(dv.runs(str(tmp_path / 'so100.runs'), tag='manual')) == 1
    with pytest.raises(dv.DexVantageError, match='Name the run store'):
        dv.runs()


def test_assistant_get_runs_tool(tmp_path, saved):
    from dexvantage.assistant import tools as T
    from dexvantage.assistant.session import FileSession
    s = FileSession(project=saved)
    out = T.call(s, 'get_runs', {})
    assert out['runs'] == [] and 'tracking is off' in out['note']
    TR.log('rig', jobs.run_study('rig', saved.snapshot()), saved)
    out = T.call(s, 'get_runs', {'study': 'rig', 'limit': 5})
    assert out['total'] == 1 and out['runs'][0]['figures']['rig.camera_count']['value'] == 0.0
    assert 'error' in T.call(s, 'get_runs', {'limit': 'x'})


def test_time_stamps_with_any_number_of_fraction_digits():
    """Python 3.10's fromisoformat reads a fraction of a second only as 3 or 6 digits; the first CI run's Python 3.10
    job failed tracking.store_round_trip on '10:00:02.5Z'. parse_utc reads 1 to 9 digits on every supported Python."""
    import datetime as dt
    from dexvantage.core.tracking.store import parse_utc, utc_text
    utc = dt.timezone.utc
    assert parse_utc('2026-11-07T10:00:02.5Z') == dt.datetime(2026, 11, 7, 10, 0, 2, 500000, tzinfo=utc)
    assert parse_utc('2026-11-07T10:00:02.125Z') == dt.datetime(2026, 11, 7, 10, 0, 2, 125000, tzinfo=utc)
    assert parse_utc('2026-11-07T10:00:02.123456789+00:00') == dt.datetime(2026, 11, 7, 10, 0, 2, 123456, tzinfo=utc)
    assert parse_utc('2026-11-07T10:00:02Z') == dt.datetime(2026, 11, 7, 10, 0, 2, tzinfo=utc)
    assert parse_utc('2026-11-07') == dt.datetime(2026, 11, 7, tzinfo=utc)
    assert parse_utc('not a time') is None
    assert utc_text(parse_utc('2026-11-07T10:00:02.5Z')) == '2026-11-07T10:00:02.500Z'
