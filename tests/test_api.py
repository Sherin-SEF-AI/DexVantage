"""The Python and notebook API (dexvantage/api.py, `import dexvantage as dv`): headless (no Qt on its path), the same
figures as the core it wraps, the documented in-place / new-object rules for factors, tables with and without pandas,
the Jupyter HTML views, and its docstring examples (doctest)."""
import doctest
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def _env(tmp_path):
    env = dict(os.environ)
    for k in ('XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_CACHE_HOME'):
        env[k] = str(tmp_path / k.lower())
    env['PYTHONPATH'] = ROOT + os.pathsep + env.get('PYTHONPATH', '')
    env.pop('QT_QPA_PLATFORM', None)
    return env


def test_api_path_imports_no_qt(tmp_path):
    code = ('import sys\n'
            'import dexvantage as dv\n'
            'import dexvantage.api\n'
            'rig = dv.sample("so100")\n'
            'res = rig.run("bandwidth")\n'
            'rig.metrics(["coverage.covered"])\n'
            'res._repr_html_(); rig._repr_html_(); res.to_frame()\n'
            'dv.studies(); dv.metrics(); dv.exporters(); dv.plugins(); dv.datasets.formats()\n'
            'bad = sorted(m for m in sys.modules if m.split(".")[0] in ("PyQt6", "PyQt5", "PySide6"))\n'
            'print("QT", bad)\n')
    out = subprocess.run([sys.executable, '-c', code], cwd=str(tmp_path), env=_env(tmp_path), capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert 'QT []' in out.stdout, out.stdout


def test_docstring_examples():
    import dexvantage.api as A
    res = doctest.testmod(A, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    assert res.failed == 0 and res.attempted >= 10, res


def test_package_reexports_the_api():
    import dexvantage as dv
    from dexvantage import api
    for name in ('open', 'sample', 'new', 'studies', 'metrics', 'exporters', 'plugins', 'datasets', 'Rig', 'Result',
                 'DexVantageError'):
        assert getattr(dv, name) is getattr(api, name), name
        assert name in dir(dv)
    with pytest.raises(AttributeError):
        dv.no_such_thing


def test_rig_wraps_the_project_and_saves(tmp_path):
    import dexvantage as dv
    rig = dv.sample('ur5e')
    assert rig.robots == rig.project.robots and len(rig.robots) == 1
    assert [s.uid for s in rig.sensors] == [s.uid for s in rig.project.variant.sensors]
    assert rig.variant is rig.project.variant and rig.variants == rig.project.variants
    rows = rig.sensor_rows()
    assert [r['name'] for r in rows] == [s.name for s in rig.sensors]
    s0 = rig.sensors[0]
    assert rows[0]['x_m'] == pytest.approx(float(s0.mount.pose.position[0]))
    with pytest.raises(dv.DexVantageError, match='never been saved'):
        dv.new('x').save()
    path = rig.save(str(tmp_path / 'cell'))
    assert path.endswith('.dexv') and os.path.isfile(path)
    back = dv.open(path)
    assert back.project.to_dict()['variants'] == rig.project.to_dict()['variants']
    assert back.path == path
    with pytest.raises(dv.DexVantageError):
        dv.Rig('not a project')


def test_factors_in_place_and_new_rigs():
    import dexvantage as dv
    from dexvantage.core import factors as F
    rig = dv.sample('so100')
    cam = rig.sensors[1]
    path = f'sensor:{cam.name}.z'
    z0 = rig.factors[path]
    assert z0 == pytest.approx(F.get(rig.project, path))
    other = rig.with_factors({path: z0 + 0.1})                 # a new Rig: rig unchanged
    assert other is not rig and other.factors[path] == pytest.approx(z0 + 0.1)
    assert rig.factors[path] == pytest.approx(z0)
    before = rig.project
    rig.factors[path] = z0 + 0.2                                 # in place: the rig's project replaced
    assert rig.factors[path] == pytest.approx(z0 + 0.2) and rig.project is not before
    assert F.get(before, path) == pytest.approx(z0)              # the old project object itself is not changed
    rig.factors.update({f'sensor:{cam.name}.dz': -0.2})
    assert rig.factors[path] == pytest.approx(z0)
    rig.project.variant.sensors[1].locked = True
    with pytest.raises(F.FactorError, match='locked'):
        rig.factors[path] = 1.0
    rig.factors.update({path: 1.0}, allow_locked=True)
    assert rig.factors[path] == pytest.approx(1.0)
    table = rig.factors.paths()
    assert any(r['path'] == path and r['locked'] for r in table)
    assert path in list(rig.factors)
    assert rig.factors.describe(path)['unit'] == 'm'
    rig.factors['setting:coverage.cube_n'] = 256
    assert rig.project.analysis.studies['coverage']['cube_n'] == 256
    assert any(r['path'] == 'setting:coverage.cube_n' for r in rig.factors.paths())
    v = rig.use_variant(rig.variants[0].name)
    assert v is not rig and v.variant.uid == rig.variant.uid
    with pytest.raises(dv.DexVantageError, match='no variant'):
        rig.use_variant('nope')


def test_run_gives_the_core_result_and_leaves_the_rig_alone():
    import dexvantage as dv
    from dexvantage.core import jobs
    rig = dv.sample('ur5e')
    res = rig.run('coverage')
    direct = jobs.run_study('coverage', rig.project.snapshot())
    assert res['summary'] == direct['summary'] and res['cameras'] == direct['cameras']
    assert isinstance(res, dict) and json.loads(json.dumps(res))['study'] == 'coverage'
    again = rig.run('coverage')
    assert again['cached'] is True                               # the API's result cache, keyed by content
    small = rig.run('coverage', cube_n=128)
    assert small['settings']['cube_n'] == 128 and 'coverage' not in rig.project.analysis.studies
    assert rig.run('depth-validity').study == 'depth_validity'  # a dash form of the id
    assert rig.run('rig-inventory').study == 'rig'              # the batch flag
    with pytest.raises(dv.DexVantageError, match='no study called'):
        rig.run('no_such_study')
    seen = []
    rig.run('coverage', cube_n=96, progress=lambda f, t: seen.append(f))
    assert seen and max(seen) <= 1.0


def test_headline_metrics_and_tables(monkeypatch):
    import dexvantage as dv
    from dexvantage.core import metrics as M
    rig = dv.sample('ur5e')
    res = rig.run('coverage')
    h = res.headline
    assert h['metric'] == 'coverage.covered' and h['unit'] == '%'
    assert h['value'] == pytest.approx(res['summary']['seen_ge1'] * 100)
    assert h['word'] == res['summary']['word'] and h['level'] == res['summary']['status']
    assert h['text'].startswith('Task volume seen: ') and h['word'] in h['text']
    ms = res.metrics()
    assert set(ms) == {m.id for m in M.METRICS.values() if m.study == 'coverage'}
    ev = rig.metrics(['coverage.covered', 'bandwidth.margin'])
    assert ev['coverage.covered']['value'] == pytest.approx(ms['coverage.covered']['value'])
    assert ev['bandwidth.margin']['value'] is None and ev['bandwidth.margin']['status'] == 'not stated'
    assert ev.figures()['coverage.covered'] == pytest.approx(h['value'])
    assert 'cameras' in res.tables()
    pd = pytest.importorskip('pandas', reason='pandas (the research extra) is not installed')
    f = res.to_frame('cameras', columns=['name', 'seen', 'fov_deg.h'])
    assert isinstance(f, pd.DataFrame) and list(f.columns) == ['name', 'seen', 'fov_deg.h']
    assert list(f['name']) == [c['name'] for c in res['cameras']]
    assert f['fov_deg.h'].tolist() == [c['fov_deg']['h'] for c in res['cameras']]
    assert isinstance(res.to_frame(), pd.DataFrame) and isinstance(rig.to_frame(), pd.DataFrame)
    with pytest.raises(dv.DexVantageError, match='has no "nope"'):
        res.to_frame('nope')
    monkeypatch.setitem(sys.modules, 'pandas', None)            # pandas not installed
    rec = res.to_frame('cameras', columns=['name', 'seen'])
    assert isinstance(rec, dv.Records) and isinstance(rec, list)
    assert rec[0] == {'name': res['cameras'][0]['name'], 'seen': res['cameras'][0]['seen']}
    assert 'pandas is not installed' in rec.note and 'dexvantage[research]' in rec.note
    assert isinstance(dv.studies().to_frame(), dv.Records)


def test_bound_and_not_stated_headlines():
    import dexvantage as dv
    rig = dv.sample('ur5e')
    ts = rig.run('time_sync').headline
    assert ts['bound'] == 'at_least' and ts['text'].split(': ')[1].startswith('≥ ')
    bw = dv.sample('so100').run('bandwidth').headline
    assert bw['value'] is None and bw['status'] == 'not stated' and 'not stated' in bw['text']


def test_html_views_escape_and_show_status():
    import dexvantage as dv
    rig = dv.sample('so100')
    rig.project.variant.sensors[0].name = '<b>odd & name</b>'
    page = rig._repr_html_()
    assert '&lt;b&gt;odd &amp; name&lt;/b&gt;' in page and '<b>odd' not in page
    assert rig.robots[0].name in page
    res = rig.run('coverage')
    h = res._repr_html_()
    assert res.headline['word'] in h and '#' in h and 'Task volume seen' in h
    assert '<table' in dv.studies()._repr_html_()
    assert '<table' in rig.metrics(['coverage.covered'])._repr_html_()
    assert repr(res).startswith('<Result coverage: Task volume seen:')


def test_export_and_check(tmp_path):
    import dexvantage as dv
    rig = dv.sample('ur5e')
    man = rig.export(str(tmp_path / 'out'), 'opencv_yaml')
    assert man['formats']['opencv_yaml']['status'] == 'ok' and man['files']
    for rel in man['files']:
        assert os.path.isfile(tmp_path / 'out' / rel)
    assert os.path.isfile(tmp_path / 'out' / 'export_manifest.json')
    seen = rig.run('coverage').headline['value']
    card = rig.check([{'name': 'Seen', 'metric': 'coverage.covered', 'op': '>=', 'threshold': seen - 1},
                      {'name': 'Seen twice', 'metric': 'coverage.covered', 'op': '>=', 'threshold': seen + 1}])
    assert isinstance(card, dv.Result) and card['verdict'] == 'fail'
    assert [i['status'] for i in card['items']] == ['pass', 'fail']
    assert card.headline['word'] == card['headline']['word'] and card.headline['text'].startswith('Readiness')
    reqs = tmp_path / 'reqs.yaml'
    reqs.write_text(f'- metric: coverage.covered\n  op: ">="\n  threshold: {seen - 1}\n', encoding='utf-8')
    assert rig.check(str(reqs))['verdict'] == 'pass'
    p = card.save(str(tmp_path / 'card.json'))
    doc = json.loads(open(p, encoding='utf-8').read())
    assert doc['verdict'] == 'fail' and 'provenance' in doc


def test_result_save_has_provenance(tmp_path):
    import dexvantage as dv
    res = dv.sample('so100').run('bandwidth')
    doc = json.loads(open(res.save(str(tmp_path / 'bw.json')), encoding='utf-8').read())
    assert doc['study'] == 'bandwidth' and 'provenance' in doc and 'cached' not in doc


def test_datasets_open(tmp_path):
    import dexvantage as dv
    pytest.importorskip('pyarrow', reason='pyarrow (the lerobot extra) is not installed')
    from dexvantage.core.datasets import episode as E
    from dexvantage.core.datasets import lerobot as L
    t = np.arange(6) / 10.0
    q = np.column_stack([np.linspace(0, 1, 6), np.linspace(1, 0, 6)])
    L.write_lerobot(str(tmp_path / 'ds'), [E.Episode('e', 0, t, q, ['a', 'b'], fps=10.0, task='move')], 'v3.0')
    with dv.datasets.open(str(tmp_path / 'ds')) as ds:
        assert ds.format == 'lerobot' and ds.version == 'v3.0'
        eps = ds.episodes()
        assert len(eps) == 1 and eps[0]['index'] == 0
        rows = ds.episode_rows(0)
        assert len(rows) == 6 and rows[-1]['a'] == pytest.approx(1.0) and rows[-1]['b'] == pytest.approx(0.0)
        assert 'LeRobot v3.0' in ds._repr_html_() and 'LeRobot v3.0' in repr(ds)
        assert ds.episode(0).task == 'move'
    assert any(r['format'] == 'lerobot' for r in dv.datasets.formats())
    with pytest.raises(dv.DexVantageError):
        dv.datasets.open(str(tmp_path / 'nothing_here'))


def test_listings():
    import dexvantage as dv
    from dexvantage.core import exporters as EX
    from dexvantage.core import jobs
    from dexvantage.core import metrics as M
    st = dv.studies()
    assert {r['id'] for r in st} == set(jobs.STUDIES)
    cov = next(r for r in st if r['id'] == 'coverage')
    assert cov['batch'] == '--coverage' and cov['plugin'] is None
    assert {r['id'] for r in dv.metrics()} == set(M.METRICS)
    assert {r['id'] for r in dv.exporters()} == set(EX.EXPORTERS)
    assert [r['name'] for r in dv.samples()][:3] == ['so100', 'ur5e', 'dexhand']
