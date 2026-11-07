"""The completeness review's additions: the research tools of part 1 from Python (dexvantage/api_research.py: dv.diff,
dv.merge, dv.bibtex, dv.figures, dv.figure, dv.export_runs), the headline of a study without a headline metric, the
plain sentence for a process pool started outside a script's main guard (core/cluster/local.check_main_guard), and
the research workflow example notebook (examples/03_research_workflow.py)."""
import json
import os
import re
import subprocess
import sys
import textwrap

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def _env(tmp_path):
    env = dict(os.environ)
    for k in ('XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_CACHE_HOME'):
        env[k] = str(tmp_path / k.lower())
    env['PYTHONPATH'] = ROOT + os.pathsep + env.get('PYTHONPATH', '')
    env.pop('QT_QPA_PLATFORM', None)
    return env


def _pair():
    import dexvantage as dv
    rig = dv.sample('so100')
    cam = next(s.name for s in rig.sensors if 'Third-person' in s.name)
    return dv, rig, rig.with_factors({f'sensor:{cam}.dz': 0.10}), cam


# ====================================================================== the API surface
def test_names_are_reexported():
    import dexvantage as dv
    from dexvantage import api, api_research
    for name in ('diff', 'merge', 'bibtex', 'figures', 'figure', 'export_runs', 'Diff', 'Merge', 'MergeConflict'):
        assert getattr(dv, name) is getattr(api, name) is getattr(api_research, name), name
        assert name in dir(dv) and name in api.__all__
    assert issubclass(dv.MergeConflict, dv.DexVantageError)


def test_new_api_path_imports_no_qt(tmp_path):
    code = textwrap.dedent('''
        import os, sys
        import dexvantage as dv
        rig = dv.sample("so100")
        other = rig.with_factors({"sensor:Third-person camera (sample placement).dz": 0.05})
        d = dv.diff(rig, other); d.text; d._repr_html_(); d.to_frame()
        dv.merge(rig, other, rig)
        dv.bibtex(rig)
        cov = rig.run("coverage")
        dv.figures(cov)
        dv.figure(cov, os.path.join(sys.argv[1], "f.svg"))
        st = dv.run_store(os.path.join(sys.argv[1], "runs")); dv.track(cov, rig, store=st)
        dv.export_runs(st, os.path.join(sys.argv[1], "x"), formats=["csv"])
        bad = sorted(m for m in sys.modules if m.split(".")[0] in ("PyQt6", "PyQt5", "PySide6"))
        print("QT", bad)
    ''')
    out = subprocess.run([sys.executable, '-c', code, str(tmp_path)], cwd=str(tmp_path), env=_env(tmp_path),
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert 'QT []' in out.stdout, out.stdout


# ====================================================================== diff
def test_diff_matches_the_batch_diff(tmp_path):
    from dexvantage.core import projdiff as D
    dv, rig, raised, cam = _pair()
    d = dv.diff(rig, raised)
    assert isinstance(d, dict) and not d.identical and d['counts'] == {'sensors': 1}
    assert dict(d) == D.diff(rig.project, raised.project)
    assert d.text == D.to_text(dict(d)) and 'mount pose moved 100.0 mm' in d.text
    assert d.html.startswith('<!doctype html>') and '<table>' in d._repr_html_()
    rows = d.to_frame()
    rows = rows.to_dict('records') if hasattr(rows, 'to_dict') else list(rows)
    assert len(rows) == 1 and rows[0]['section'] == 'sensors' and rows[0]['op'] == 'changed'
    assert repr(d) == "<Diff 'SO-100 sample' → 'SO-100 sample': 1 change>"
    assert dv.diff(rig, rig.copy()).identical
    # paths work as well as rigs, and save() writes the batch JSON with provenance
    a, b = rig.save(str(tmp_path / 'a.dexv')), raised.save(str(tmp_path / 'b.dexv'))
    dp = dv.diff(a, b)
    assert dp['changes'] == d['changes'] and dp['a']['file'] == 'a.dexv'
    doc = json.load(open(dp.save(str(tmp_path / 'diff.json')), encoding='utf-8'))
    assert doc['schema'] == D.SCHEMA and 'provenance' in doc
    with pytest.raises(dv.DexVantageError, match='tables "changes" and "metrics"'):
        d.to_frame('nope')


def test_diff_metric_deltas_equal_the_studies():
    dv, rig, raised, _cam = _pair()
    d = dv.diff(rig, raised, metrics='coverage.covered')
    (m,) = d['metrics']
    a, b = rig.metrics(['coverage.covered'])['coverage.covered']['value'], \
        raised.metrics(['coverage.covered'])['coverage.covered']['value']
    assert m['id'] == 'coverage.covered' and m['a'] == pytest.approx(a) and m['b'] == pytest.approx(b)
    assert m['delta'] == pytest.approx(b - a) and m['verdict'] in ('better', 'worse', 'same')
    assert 'Figures of merit' in d.text
    with pytest.raises(dv.DexVantageError, match='Unknown metric no.such'):
        dv.diff(rig, raised, metrics=['no.such'])


def test_diff_refuses_what_is_not_a_project(tmp_path):
    import dexvantage as dv
    rig = dv.sample('so100')
    with pytest.raises(dv.DexVantageError, match='second project is a Rig'):
        dv.diff(rig, 42)
    with pytest.raises(dv.DexVantageError):
        dv.diff(rig, str(tmp_path / 'missing.dexv'))


# ====================================================================== merge
def test_merge_two_edits_and_a_conflict():
    dv, rig, raised, cam = _pair()
    other = next(s.name for s in rig.sensors if s.name != cam)
    turned = rig.with_factors({f'sensor:{other}.dz': 0.02})
    m = dv.merge(rig, raised, turned)
    assert isinstance(m, dv.Merge) and isinstance(m.rig, dv.Rig)
    assert m.rig.factors[f'sensor:{cam}.z'] == pytest.approx(raised.factors[f'sensor:{cam}.z'])
    assert m.rig.factors[f'sensor:{other}.z'] == pytest.approx(turned.factors[f'sensor:{other}.z'])
    assert m.from_ours and m.from_theirs and 'of ours' in repr(m)
    lower = rig.with_factors({f'sensor:{cam}.dz': -0.05})
    with pytest.raises(dv.MergeConflict) as ei:
        dv.merge(rig, raised, lower)
    assert ei.value.conflicts and 'cannot be merged' in str(ei.value)


# ====================================================================== citations and figures
def test_bibtex_equals_core_cite(tmp_path):
    import dexvantage as dv
    from dexvantage.core import cite as C
    rig = dv.sample('so100')
    text = dv.bibtex(rig, str(tmp_path / 'refs.bib'), references='none')
    assert text == C.bibtex(rig.project, 'none')
    assert open(tmp_path / 'refs.bib', encoding='utf-8').read() == text
    assert text.count('\n@') >= 2 and '@software{' in text
    with pytest.raises(dv.DexVantageError, match='"checks", "all" or "none"'):
        dv.bibtex(rig, references='some')


def test_figures_of_a_result_are_the_paper_kits(tmp_path):
    import dexvantage as dv
    from dexvantage.core import publication as PB
    rig = dv.sample('so100')
    cov = rig.run('coverage')
    figs = dv.figures(cov)
    kit = PB.figure_specs({'coverage': dict(cov)})
    assert [f['id'] for f in figs] == [k[0] for k in kit] and figs[0]['spec'] == kit[0][1]
    svg = dv.figure(cov, str(tmp_path / 'c.svg'))
    assert svg['format'] == 'svg' and svg['width_mm'] == pytest.approx(88.9)
    text = open(svg['path'], encoding='utf-8').read()
    assert text.startswith('<svg') and 'dexvantage.figure/1' in text
    # a spec, a dv.figures item and a spec file give the same picture
    spec = figs[0]['spec']
    (tmp_path / 'spec.json').write_text(json.dumps(spec), encoding='utf-8')
    for src in (figs[0], str(tmp_path / 'spec.json')):
        r = dv.figure(src, str(tmp_path / 'd.svg'), width='double')
        assert r['width_mm'] == pytest.approx(182.0)
    with pytest.raises(dv.DexVantageError, match='index 5'):
        dv.figure(cov, str(tmp_path / 'e.svg'), index=5)
    with pytest.raises(dv.DexVantageError, match='palette'):
        dv.figure(cov, str(tmp_path / 'e.svg'), palette='rainbow')
    bw = rig.run('bandwidth')
    if not dv.figures(bw):
        with pytest.raises(dv.DexVantageError, match='no chart to draw'):
            dv.figure(bw, str(tmp_path / 'e.svg'))


# ====================================================================== tracked runs
def test_export_runs_writes_every_format(tmp_path):
    import dexvantage as dv
    rig = dv.sample('so100')
    st = dv.run_store(str(tmp_path / 'runs'))
    with pytest.raises(dv.DexVantageError, match='no run store'):
        dv.export_runs(st, str(tmp_path / 'x'))
    dv.track(rig.run('coverage'), rig, store=st)
    dv.track(rig.run('bandwidth'), rig, store=st)
    r = dv.export_runs(st, str(tmp_path / 'out'))
    assert {f: v['status'] for f, v in r['formats'].items()} == {f: 'ok' for f in ('csv', 'parquet', 'mlflow',
                                                                                  'wandb')} or \
        r['formats']['parquet']['status'] != 'ok'                         # pyarrow may be absent: said so
    assert os.path.isfile(tmp_path / 'out' / 'runs.csv') and os.path.isdir(tmp_path / 'out' / 'mlruns')
    assert len(r['runs']) == 2 if isinstance(r['runs'], list) else r['runs'] == 2
    only = dv.export_runs(st, str(tmp_path / 'cov'), formats='csv', study='coverage')
    assert (len(only['runs']) if isinstance(only['runs'], list) else only['runs']) == 1


# ====================================================================== headlines
def test_headline_without_a_metric_shows_the_unit_and_quantile():
    from dexvantage.api import Result
    doc = {'study': 'tolerance', 'headline': {'value': 94.344085, 'unit': '%', 'quantile': 'P5', 'status': 'warning',
                                              'word': 'no requirement', 'text': 'Task volume seen ...'}}
    h = Result(doc).headline
    assert h['value'] == 94.344085 and h['unit'] == '%'
    assert h['text'].startswith('Monte Carlo tolerances: P5 94.3 % (no requirement). ')
    h = Result({'study': 'requirements', 'headline': {'value': '0 / 0', 'word': 'no requirements'}}).headline
    assert h['text'].startswith('Readiness scorecard: 0 / 0 (no requirements)')


def test_p_value_reads_as_p_equals():
    import dexvantage as dv
    r = dv.stats.compare_rates([('A', 14, 20), ('B', 19, 20)])
    assert re.search(r': p = 0\.09\d* \(no clear difference\)', r.headline['text']), r.headline['text']


# ====================================================================== the main guard
def test_guard_line_reads_the_script(tmp_path):
    from dexvantage.core.cluster.local import _guard_line
    p = tmp_path / 's.py'
    p.write_text('import x\nx.run()\n\n\ndef main():\n    x.run()\n\n\nif __name__ == "__main__":\n    main()\n'
                 '    x.run()\nif "__main__" == __name__:\n    x.run()\n', encoding='utf-8')
    assert _guard_line(str(p), 2) is False and _guard_line(str(p), 6) is False
    assert _guard_line(str(p), 10) is True and _guard_line(str(p), 11) is True and _guard_line(str(p), 13) is True
    assert _guard_line(str(tmp_path / 'missing.py'), 1) is None


def test_check_main_guard_skips_pytest_and_modules():
    from dexvantage.core.cluster.local import check_main_guard
    check_main_guard()                    # pytest's __main__ is a module run by name: spawn does not re-run it


def test_unguarded_script_gets_one_sentence(tmp_path):
    s = tmp_path / 'sweep.py'
    s.write_text('import dexvantage as dv\nfrom dexvantage.core.cluster.local import check_main_guard\n'
                 'try:\n    check_main_guard("A pool of 2 workers")\nexcept dv.DexVantageError as e:\n'
                 '    print("REFUSED", e)\n', encoding='utf-8')
    out = subprocess.run([sys.executable, str(s)], cwd=str(tmp_path), env=_env(tmp_path), capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert 'REFUSED A pool of 2 workers starts worker processes' in out.stdout
    assert "(line 4 of sweep.py is outside it)" in out.stdout and 'Traceback' not in out.stdout + out.stderr
    g = tmp_path / 'guarded.py'
    g.write_text('import dexvantage as dv\nfrom dexvantage.core.cluster.local import check_main_guard\n'
                 'if __name__ == "__main__":\n    check_main_guard()\n    print("OK")\n', encoding='utf-8')
    out = subprocess.run([sys.executable, str(g)], cwd=str(tmp_path), env=_env(tmp_path), capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0 and 'OK' in out.stdout, out.stderr[-2000:]


# ====================================================================== the example
def test_research_notebook_is_percent_format():
    text = open(os.path.join(ROOT, 'examples', '03_research_workflow.py'), encoding='utf-8').read()
    assert text.startswith('# ---\n# jupyter:\n#   jupytext:\n') and 'format_name: percent' in text
    cells = re.findall(r'^# %%( \[markdown\])?$', text, flags=re.M)
    assert sum(1 for c in cells if c) >= 8 and sum(1 for c in cells if not c) >= 8
    compile(text, '03_research_workflow.py', 'exec')


def test_research_notebook_runs(tmp_path):
    env = _env(tmp_path)
    env['DEXVANTAGE_EXAMPLE_OUT'] = str(tmp_path / 'out')
    out = subprocess.run([sys.executable, os.path.join(ROOT, 'examples', '03_research_workflow.py')],
                         cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-3000:]
    o = out.stdout
    assert re.search(r"^\{'csv': 'ok', 'parquet': '\w+', 'mlflow': 'ok', 'wandb': 'ok'\}$", o, flags=re.M), o
    assert 'Episodes per policy for 70 % against 95 %: ' in o and 'no clear difference' in o
    assert re.search(r'^Campaign episodes recorded: \d+', o, flags=re.M), o
    assert re.search(r'^Monte Carlo tolerances: P5 [0-9.]+ % ', o, flags=re.M), o
    assert re.search(r'^Largest geometric gap to the reference cell: ', o, flags=re.M), o
    assert 'mount pose moved 100.0 mm' in o and 'Figures of merit (a → b):' in o
    assert re.search(r'^ok \d+ files$', o, flags=re.M) and re.search(r'^\d+ BibTeX entries$', o, flags=re.M), o
    assert 'coverage.svg 88.9 × ' in o
    for f in ('runs_export/runs.csv', 'paper/paper_kit/main.tex', 'paper/paper_kit/refs.bib', 'refs.bib',
              'coverage.svg', 'so100_raised.dexv.d/rig.json'):
        assert os.path.exists(tmp_path / 'out' / f), f
