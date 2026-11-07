"""The publication kit (core/publication.py, the 'paper_kit' exporter, dv.publication, the assistant's
get_methods_paragraph): every number of the methods paragraph found in the source data, citation keys equal to
refs.bib, LaTeX structure (compiled when an engine is installed), table numbers equal to the results, figures at
both widths, the run plan that regenerates the kit, and the validation checks."""
import json
import math
import os
import re
import shutil
import subprocess
import sys

import pytest

from dexvantage.core import exporters as X
from dexvantage.core import jobs
from dexvantage.core import publication as PB

CHEAP = ('rig', 'bandwidth', 'coverage', 'time_sync', 'motion_blur', 'depth_validity')
GRASP = {'schema': 'dexvantage.grasp_visibility/1', 'study': 'grasp_visibility',
         'summary': {'cvs_pct': 96.5, 'ci_low': 0.931, 'ci_high': 0.987, 'ci_level': 0.95,
                     'ci_method': 'percentile bootstrap', 'grasps': 40, 'feasible': 38, 'both_fingertips_rate': 0.9}}


def _sample(name):
    from dexvantage.core.robot.samples import sample_project
    return sample_project(name)


def _results(p, studies=CHEAP):
    out = {}
    for sid in studies:
        try:
            out[sid] = jobs.run_study(sid, p.snapshot())
        except jobs.StudyError:
            pass
    return out


# ====================================================================== the numbers of the paragraph
NUM = re.compile(r'(?<![A-Za-z0-9_.\\])-?[0-9]+(?:\.[0-9]+)*(?![A-Za-z0-9_]|\.[0-9])')


def _paragraph_numbers(tex):
    """Every number of the paragraph, found here: comments, the macro definition, citation keys and labels out."""
    t = '\n'.join(re.sub(r'(?<!\\)%.*$', '', ln) for ln in tex.split('\n'))
    t = re.sub(r'^\\providecommand.*$', '', t, flags=re.M)
    t = re.sub(r'\\(?:cite|ref|label)\{[^{}]*\}', ' ', t)
    return NUM.findall(t)


def _pool(p, results):
    """Numbers, words of strings and list lengths of the source data, collected here from the project's JSON, the
    library files on disk, the results, the versions and the check registry."""
    import importlib.metadata as md
    import platform

    from dexvantage import __version__
    from dexvantage.core import library as LB
    from dexvantage.validation import suite as VS
    nums, strs, lens = [], set(), set()

    def walk(o):
        if isinstance(o, dict):
            lens.add(len(o))
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            lens.add(len(o))
            for v in o:
                walk(v)
        elif isinstance(o, bool):
            pass
        elif isinstance(o, (int, float)):
            nums.append(float(o))
        elif isinstance(o, str):
            strs.add(o)
    walk(p.to_dict())
    walk(results)
    for mid in {s.model_id for v in p.variants for s in v.sensors if s.model_id}:
        with open(LB.get(mid).path, encoding='utf-8') as fh:
            walk(json.load(fh))
    walk([list(LB.DEFAULT_RESOLUTION), LB.DEFAULT_HFOV_DEG, LB.DEFAULT_VFOV_DEG])
    strs |= {__version__, platform.python_version(), md.version('mujoco')}
    VS.load_checks()
    per = {}
    for sp in VS.CHECKS:
        per[sp.model] = per.get(sp.model, 0) + 1
    nums += [float(v) for v in per.values()]
    words = set()
    for s in strs:
        words |= set(NUM.findall(s))
    return nums, words, lens


def _found(tok, nums, words, lens):
    if tok in words:
        return True
    if tok.count('.') > 1:
        return False
    t = float(tok)
    d = len(tok.split('.')[1]) if '.' in tok else 0
    if d == 0 and t in {float(x) for x in lens}:
        return True
    return any(abs(v - t) <= 0.5 * 10 ** -d * (1 + 1e-9) + 1e-12 for v in nums)


@pytest.mark.parametrize('name,si', [('so100', False), ('ur5e', True)])
def test_every_number_of_the_methods_paragraph_is_in_the_source_data(name, si):
    p = _sample(name)
    res = _results(p)
    kit = PB.build(p, {'results': res, 'studies': [], 'siunitx': si})
    tex = kit['parts']['methods']
    toks = _paragraph_numbers(tex)
    assert len(toks) >= 25
    nums, words, lens = _pool(p, res)
    missing = [t for t in toks if not _found(t, nums, words, lens)]
    assert missing == [], missing
    # and the kit's own trace agrees: every number has an entry whose source holds it
    assert kit['checks']['numbers']['problems'] == []
    assert kit['checks']['numbers']['numbers'] == len(toks)
    assert kit['status']['word'] == 'ready', kit['status']
    # what the paragraph must describe
    for s in p.variant.sensors:
        assert PB.tex(s.name) in tex
    assert 'MuJoCo' in tex and r'\cite{dexvantage}' in tex
    if name == 'ur5e':
        assert 'Intel RealSense Depth Camera D405' in tex and r'\qty{0.018}{\metre}' in tex
        assert r'\num{1280}~$\times$~\num{720}~px' in tex and 'fps' in tex and r'\degree' in tex
        assert 'wrist\\_3\\_link' in tex and 'static frame \\texttt{cage}' in tex
    else:
        assert 'default camera model' in tex and 'DexVantage defaults, not a device' in tex


def test_a_number_without_source_is_caught():
    p = _sample('so100')
    res = _results(p, ('rig',))
    kit = PB.build(p, {'results': res, 'studies': []})
    src = kit['_source']
    bad = kit['parts']['methods'].replace('version 3', 'version 3 and 123.456', 1) + ' 77.7'
    v = PB.verify_numbers(bad, kit['trace'], src)
    assert '77.7 has no source' in v['problems']
    # a traced number changed in the source no longer verifies
    tr = [dict(e) for e in kit['trace']]
    k = next(i for i, e in enumerate(tr) if e['kind'] == 'number' and e['path'][:2] == ['project', 'variants'])
    tr[k]['text'] = '9.876'
    assert PB.verify_numbers(kit['parts']['methods'], tr, src)['problems']
    nums, words, lens = _pool(p, res)
    assert not _found('123.456', nums, words, lens) and _found('0.360', nums, words, lens)


# ====================================================================== citations, LaTeX, tables
def _kit_folder(tmp_path, p, res, **opts):
    man = X.export(p, str(tmp_path), ['paper_kit'], {'paper_kit': {'results': res, 'studies': [], **opts}})
    assert man['formats']['paper_kit']['status'] == 'ok', man['formats']['paper_kit']
    return str(tmp_path / 'paper_kit'), man


def test_bibtex_keys_used_equal_keys_defined(tmp_path):
    from dexvantage.validation.checks_versioning import parse_bibtex
    p = _sample('ur5e')
    folder, _ = _kit_folder(tmp_path, p, _results(p))
    cited = set()
    for root, _d, files in os.walk(folder):
        for f in files:
            if f.endswith('.tex'):
                with open(os.path.join(root, f), encoding='utf-8') as fh:
                    for m in re.finditer(r'\\cite\{([^{}]*)\}', fh.read()):
                        cited |= {k.strip() for k in m.group(1).split(',')}
    with open(os.path.join(folder, 'refs.bib'), encoding='utf-8') as fh:
        defined = {k for _t, k, _f in parse_bibtex(fh.read())}
    assert cited == defined and len(cited) >= 8
    assert {'dexvantage', 'menagerie2022github', 'intel-realsense-d405', 'mujoco_todorov2012'} <= cited


def _compile(folder, engine):
    cmds = [[engine, '-interaction=nonstopmode', 'main.tex']] if engine != 'tectonic' else [[engine, 'main.tex']]
    if engine != 'tectonic' and shutil.which('bibtex'):
        cmds = cmds + [['bibtex', 'main']] + cmds * 2
    for c in cmds:
        r = subprocess.run(c, cwd=folder, capture_output=True, text=True, timeout=240)
        assert r.returncode == 0 or c[0] == 'bibtex', r.stdout[-3000:]
    assert os.path.isfile(os.path.join(folder, 'main.pdf'))


@pytest.mark.parametrize('si', [False, True])
def test_latex_structure_and_compile_when_an_engine_is_installed(tmp_path, si):
    from dexvantage.validation.checks_publication import structure_problems
    p = _sample('so100')
    res = _results(p)
    res['grasp_visibility'] = GRASP
    folder, man = _kit_folder(tmp_path, p, res, siunitx=si)
    texs = []
    for root, _d, files in os.walk(folder):
        texs += [os.path.join(root, f) for f in files if f.endswith('.tex')]
    assert len(texs) >= 5
    for f in texs:
        with open(f, encoding='utf-8') as fh:
            text = fh.read()
        assert structure_problems(text, folder) == [], f
        assert PB.latex_problems(text) == [], f
        assert r'\hline' not in text and not re.search(r'\\begin\{tabular\}\{[^}]*\|', text)
    with open(os.path.join(folder, 'main.tex'), encoding='utf-8') as fh:
        main = fh.read()
    assert ('\\usepackage{siunitx}' in main) == si and '\\usepackage{booktabs}' in main
    assert PB.missing_inputs(folder, main) == []
    engine = PB.latex_engine()
    if engine is None:
        pytest.skip('No LaTeX engine (pdflatex, lualatex, tectonic) on PATH: the structural check ran, the compile '
                    'did not.')
    _compile(folder, engine)


def test_the_structural_check_finds_broken_tex():
    from dexvantage.validation.checks_publication import structure_problems
    assert PB.latex_problems(r'\begin{table}\begin{tabular}{l}x\end{table}')
    assert PB.latex_problems('a {b')
    assert PB.latex_problems(r'a \{ b \} %{ comment') == []
    assert PB.latex_problems('\\begin{verbatim}\n{ not counted \\end{x}\n\\end{verbatim}') == []
    assert structure_problems('\\begin{tabular}{l|r}\n\\hline\n\\end{tabular}', '.')
    assert structure_problems('\\input{nowhere}', '.') == ['line 1: nowhere is not in the kit']


def test_table_numbers_equal_the_results(tmp_path):
    from dexvantage.core import metrics as M
    p = _sample('ur5e')
    res = _results(p)
    res['grasp_visibility'] = GRASP
    kit = PB.build(p, {'results': res, 'studies': []})
    assert kit['checks']['tables']['problems'] == [] and kit['checks']['tables']['numbers'] >= 25
    table = kit['parts']['results']
    # every named figure the results state is in the table, at the value metrics.read gives
    M.load_metrics()
    seen = 0
    for line in table.split('\n'):
        m = re.search(r'% dv-src: (\[.*\])$', line)
        if not m:
            continue
        ref = json.loads(m.group(1))[0]
        if ref[0] != 'metric':
            continue
        v = M.read(M.METRICS[ref[2]], res[ref[1]])['value']
        cell = re.search(r'\\dvnum\{([^{}]*)\}', line).group(1)
        assert math.isclose(float(cell), v, rel_tol=5e-4, abs_tol=5e-4), (ref, cell, v)
        seen += 1
    assert seen >= 12
    # the interval column: the grasp result's bootstrap interval in percent with its level
    row = next(ln for ln in table.split('\n') if 'Contact visibility &' in ln)
    assert r'[\dvnum{93.1}, \dvnum{98.7}]' in row and r'\dvnum{95}\,\%' in row and 'percentile bootstrap' in row
    # a table number that differs from its source is caught
    bad = {'results': table.replace(r'\dvnum{96.5}', r'\dvnum{97.5}')}
    assert PB.verify_tables(bad, kit['_source'])['problems']
    # the sensor table: the D405's stream, rate and FOV from its library file
    st = kit['parts']['sensors']
    assert r'\dvnum{1280}$\times$\dvnum{720}' in st and r'\dvnum{84}$\times$\dvnum{58}' in st


def _experiment_doc():
    return {'schema': 'dexvantage.experiments/1', 'study': 'experiments', 'name': 'pitch and height',
            'method': 'saltelli', 'method_label': 'Saltelli (Sobol indices)',
            'factors': [{'path': 'sensor:Overhead D435.z', 'title': 'Overhead D435 height', 'unit': 'm',
                         'kind': 'continuous', 'lo': 0.8, 'hi': 1.2, 'current': 1.0},
                        {'path': 'sensor:Overhead D435.pitch', 'title': 'Overhead D435 pitch', 'unit': '°',
                         'kind': 'continuous', 'lo': 60.0, 'hi': 90.0, 'current': 90.0}],
            'responses': [{'id': 'coverage.covered', 'title': 'Task volume seen', 'unit': '%', 'better': 'higher'}],
            'runs': [{'run': i + 1} for i in range(16)],
            'analysis': {'coverage.covered': {'n': 16, 'missing': 0, 'mean': 91.2, 'sd': 2.31, 'min': 86.0,
                                              'max': 95.4,
                                              'sobol': {'factors': ['sensor:Overhead D435.z',
                                                                    'sensor:Overhead D435.pitch'],
                                                        'S1': [0.31, 0.52], 'ST': [0.4, 0.66],
                                                        'S1_ci': [[0.1, 0.5], [0.3, 0.7]],
                                                        'ST_ci': [[0.2, 0.6], [0.45, 0.85]], 'N': 4,
                                                        'confidence': 0.95}}}}


def _tolerance_doc():
    return {'schema': 'dexvantage.tolerance/1', 'study': 'tolerance', 'n': 64, 'done': 64, 'seed': 7, 'scale': 1.0,
            'distribution': 'uniform',
            'metrics': {'coverage.covered': {'title': 'Task volume seen', 'unit': '%', 'better': 'higher',
                                             'nominal': 91.9, 'valid': 64, 'p5': 89.25, 'p50': 91.7, 'p95': 93.1,
                                             'mean': 91.5, 'std': 1.2}}}


def test_experiment_tolerance_scorecard_and_variant_tables(tmp_path):
    from dexvantage.core import requirements as RQ
    from dexvantage.core.project import Variant
    p = _sample('ur5e')
    p.requirements = [RQ.clean({'name': 'Two cameras at most', 'metric': 'rig.camera_count', 'op': '<=',
                                'threshold': 2}),
                      RQ.clean({'metric': 'coverage.covered', 'op': '>=', 'threshold': 90})]
    v2 = Variant.from_dict(p.variant.to_dict())
    v2.uid, v2.name = 'v2uid00000', 'Overhead only'
    v2.sensors = v2.sensors[1:]
    p.variants.append(v2)
    res = _results(p, ('rig', 'coverage'))
    res['experiments'] = _experiment_doc()
    res['tolerance'] = _tolerance_doc()
    kit = PB.build(p, {'results': res, 'studies': []})
    assert kit['status']['word'] == 'ready', kit['status']
    ex = kit['parts']['experiments']
    assert r'\dvnum{0.4} [\dvnum{0.2}, \dvnum{0.6}]' in ex and r'\dvnum{91.2}' in ex and '95\\,\\% bootstrap' in ex
    assert r'[\dvnum{89.25}, \dvnum{93.1}]' in kit['parts']['results']
    sc = kit['parts']['scorecard']
    assert 'Two cameras at most' in sc and 'pass' in sc and r'\dvnum{2}' in sc
    assert 'Overhead only' in kit['parts']['variants']
    meth = kit['parts']['methods']
    assert r'drew \dvnum{64} perturbed rigs (seed \dvnum{7}' in meth and r'ran \dvnum{16} runs of a' in meth
    assert r'one of \dvnum{2} rig variants' in meth and 'its \\dvnum{2} requirements' in meth
    # figures: the experiment's Sobol indices
    assert any(fid.startswith('experiments-sobol') for fid, _s, _sid in kit['figures'])


def test_scorecard_waits_for_results_in_preview():
    from dexvantage.core import requirements as RQ
    p = _sample('so100')
    p.requirements = [RQ.clean({'metric': 'sync.skew_ms', 'op': '<', 'threshold': 2})]
    kit = PB.build(p, {'results': _results(p, ('rig',)), 'studies': []})
    assert kit['scorecard_missing'] == ['time_sync'] and 'needs the results of time' in kit['parts']['scorecard']


# ====================================================================== the export, figures and the run plan
def test_export_writes_every_file_with_figures_at_both_widths(tmp_path):
    p = _sample('ur5e')
    res = _results(p)
    folder, man = _kit_folder(tmp_path, p, res)
    files = man['formats']['paper_kit']['files']
    for rel in ('main.tex', 'methods.tex', 'appendix.tex', 'refs.bib', 'provenance.json', 'reproduce/run_plan.json',
                'tables/sensors.tex', 'tables/results.tex'):
        assert f'paper_kit/{rel}' in files and f'paper_kit/{rel}' in man['files'], rel
    figs = [f for f in files if '/figures/' in f]
    ids = {re.sub(r'-(single|double)\.(svg|pdf)$', '', os.path.basename(f)) for f in figs}
    assert ids
    for fid in ids:
        for w in ('single', 'double'):
            for ext in ('svg', 'pdf'):
                assert f'paper_kit/figures/{fid}-{w}.{ext}' in figs, (fid, w, ext)
    with open(os.path.join(folder, 'figures', f'{sorted(ids)[0]}-double.pdf'), 'rb') as fh:
        data = fh.read()
    box = [float(x) for x in re.search(rb'/MediaBox\s*\[([^\]]+)\]', data).group(1).split()]
    assert abs(box[2] - box[0] - 182.0 / 25.4 * 72) < 1.0
    with open(os.path.join(folder, 'provenance.json'), encoding='utf-8') as fh:
        prov = json.load(fh)
    assert prov['schema'] == PB.SCHEMA and prov['provenance']['schema'] == 'dexvantage.provenance/1'
    assert prov['status']['word'] == 'ready' and prov['trace'] and 'main.tex' in prov['files']
    with open(os.path.join(folder, 'main.tex'), encoding='utf-8') as fh:
        main = fh.read()
    assert r'\input{methods}' in main and r'\bibliography{refs}' in main and r'\input{appendix}' in main
    assert re.search(r'\\includegraphics\[width=\\columnwidth\]\{figures/[^}]+-single\.pdf\}', main)
    text = open(os.path.join(folder, 'appendix.tex'), encoding='utf-8').read()
    assert 'run\\_plan.json' in text and ('Random seeds' in text or 'draws random numbers' in text)
    assert 'mujoco' in text
    # no personal path in any file of the kit
    for root, _d, fs in os.walk(folder):
        for f in fs:
            if f.endswith(('.tex', '.bib', '.json', '.svg')):
                body = open(os.path.join(root, f), encoding='utf-8').read()
                assert '/home/' not in body and str(tmp_path) not in body, f


def test_sections_and_options_choose_what_is_written(tmp_path):
    p = _sample('so100')
    folder, man = _kit_folder(tmp_path, p, _results(p, ('rig',)), sections=['methods', 'sensors'],
                              widths=['single'], pdf_figures=False, bibliography='biblatex', title='A rig')
    files = man['formats']['paper_kit']['files']
    assert 'paper_kit/tables/results.tex' not in files and 'paper_kit/appendix.tex' not in files
    assert not any('/figures/' in f for f in files)
    main = open(os.path.join(folder, 'main.tex'), encoding='utf-8').read()
    assert r'\addbibresource{refs.bib}' in main and r'\printbibliography' in main and r'\title{A rig}' in main
    with pytest.raises(jobs.StudyError, match='sections'):
        PB.settings_of(p, {'sections': ['methods', 'poems']})
    with pytest.raises(jobs.StudyError, match='widths'):
        PB.settings_of(p, {'widths': []})
    # settings kept in the project apply to a later export
    p.analysis.studies['publication'] = {'sections': ['methods'], 'siunitx': True}
    kit = PB.build(p, {'results': {}, 'studies': []})
    assert kit['sections'] == ['methods'] and kit['siunitx'] and '\\num{' in kit['parts']['methods']


def test_the_run_plan_regenerates_the_kit(tmp_path):
    from dexvantage.core import run_plan as RP
    p = _sample('so100')
    folder, _ = _kit_folder(tmp_path / 'first', p, _results(p, ('rig', 'bandwidth', 'time_sync')))
    plan = RP.load(os.path.join(folder, 'reproduce', 'run_plan.json'))
    # the steps follow the study registry's order, which depends on which modules were imported first: compare sets
    assert sorted(s['study'] for s in plan['steps'] if s['kind'] == 'study') == ['bandwidth', 'rig', 'time_sync']
    out = tmp_path / 'again'
    summary, code = RP.run(p, plan, str(out))
    assert code == 0, summary
    with open(out / 'paper_kit' / 'methods.tex', encoding='utf-8') as fh:
        again = fh.read()
    with open(os.path.join(folder, 'methods.tex'), encoding='utf-8') as fh:
        first = fh.read()
    assert _paragraph_numbers(again) == _paragraph_numbers(first)
    with open(out / 'paper_kit' / 'tables' / 'results.tex', encoding='utf-8') as fh:
        assert fh.read() == open(os.path.join(folder, 'tables', 'results.tex'), encoding='utf-8').read()


def test_validation_report_gives_each_model_its_status(tmp_path):
    p = _sample('so100')
    val = {'schema': 'dexvantage.validation/1', 'created_utc': '2026-11-07T10:39:23Z',
           'models': [{'id': 'bandwidth', 'passed': 3, 'checks': [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}]}]}
    path = tmp_path / 'validation.json'
    path.write_text(json.dumps(val))
    kit = PB.build(p, {'results': _results(p, ('bandwidth',)), 'studies': [], 'validation': str(path)})
    assert r'passed \dvnum{3} of \dvnum{3} checks' in kit['parts']['methods']
    assert kit['status']['word'] == 'ready'
    path.write_text('{"schema": "other"}')
    with pytest.raises(jobs.StudyError, match='not a DexVantage validation report'):
        PB.build(p, {'results': {}, 'studies': [], 'validation': str(path)})


# ====================================================================== API, assistant, validation
def test_python_api_is_headless():
    code = ('import sys, dexvantage as dv\n'
            'rig = dv.sample("so100")\n'
            'res = {"rig": rig.run("rig"), "bandwidth": rig.run("bandwidth")}\n'
            'm = dv.publication.methods(rig, res)\n'
            'k = dv.publication.kit(rig, res)\n'
            'assert "SO-ARM100" in m and k["status"]["word"] == "ready", k["status"]\n'
            'assert set(dv.publication.tables(rig, res)) >= {"sensors", "results"}\n'
            'assert not any(n.startswith("PyQt6") for n in sys.modules), "Qt imported"\n'
            'print("ok")\n')
    env = dict(os.environ)
    env.pop('QT_QPA_PLATFORM', None)
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=180, env=env)
    assert r.returncode == 0 and 'ok' in r.stdout, r.stdout + r.stderr


def test_api_export(tmp_path):
    import dexvantage as dv
    rig = dv.sample('so100')
    man = dv.publication.export(rig, tmp_path, {'rig': rig.run('rig')}, widths='single')
    assert man['formats']['paper_kit']['status'] == 'ok'
    with pytest.raises(dv.DexVantageError):
        dv.publication.kit('not a rig')


def test_assistant_reads_the_methods_paragraph(tmp_path):
    from dexvantage.assistant import tools as T
    from dexvantage.assistant.session import FileSession
    p = _sample('so100')
    path = str(tmp_path / 'so100.dexv')
    p.save(path)
    s = FileSession(path=path)
    out = T.call(s, 'get_methods_paragraph', {'studies': ['rig']})
    assert 'error' not in out, out
    assert out['status'] == 'ready' and out['studies'] == ['rig'] and 'dexvantage' in out['citations']
    assert out['numbers'] == out['numbers_traced'] > 10
    json.dumps(out)
    assert 'error' in T.call(s, 'get_methods_paragraph', {'studies': ['grasp_visibility'], 'x': 1})


def test_validation_checks_pass():
    from dexvantage.validation import suite as VS
    VS.load_checks()
    ctx = VS.Ctx(None, None, None, VS.ValidationSettings(quick=True))
    try:
        res = [VS.run_one(sp, ctx) for sp in VS.CHECKS if sp.model == 'publication']
    finally:
        ctx.close()
    assert len(res) == 4
    for c in res:
        assert c.passed, (c.id, c.error, c.detail, c.exception)
    assert next(c for c in res if c.id == 'publication.table_numbers').detail['cells'] >= 40


# ====================================================================== the page
def _wait_page(app, page, timeout=60):
    import time

    from conftest import pump
    t = time.time()
    while page._busy and time.time() - t < timeout:
        pump(app, 0.05)
    pump(app, 0.05)


def test_publication_page_preview_apply_export(app, win, tmp_path):
    import time

    from conftest import pump
    win.store.set_project(_sample('so100'))
    win.ctx.analysis.wait(60)
    for sid in ('rig', 'bandwidth'):
        win.ctx.analysis.run(sid)
        win.ctx.analysis.wait(60)
    pump(app, 0.1)
    assert 'publication' in win.pages
    page = win.pages['publication']
    a = win.actions_by_id['page_publication']
    assert a.text() == 'Paper kit…' and a in win.menus['Export'].actions()
    win.open_page('publication')
    assert page.head.status.text.text() == 'not built'
    page.run()
    _wait_page(app, page)
    assert page.kit is not None, page.note.text()
    assert page.head.status.text.text() == 'ready', page.head.state
    n = page.kit['status']['numbers']
    assert page.head.figure.text() == f'{n} / {n}' and n > 10
    assert {'rig', 'bandwidth'} <= set(page.kit['results'])
    assert 'SO-ARM100' in page.text.toPlainText()
    page.part.setCurrentIndex(page.part.findData('refs.bib'))
    assert page.text.toPlainText().lstrip().startswith('% BibTeX')
    # Apply keeps the choices in the project, one undo step, within the UI budget
    assert not page.b_apply.isEnabled()
    page.c_si.setChecked(True)
    page.sec_checks['appendix'].setChecked(False)
    assert page.b_apply.isEnabled()
    u = win.store.undo
    n0 = u.index()
    t0 = time.perf_counter()
    page.apply()
    assert (time.perf_counter() - t0) < 0.1
    assert u.index() == n0 + 1
    st = win.store.project.analysis.studies['publication']
    assert st['siunitx'] is True and 'appendix' not in st['sections'] and not page.b_apply.isEnabled()
    u.undo()
    pump(app, 0.05)
    assert 'publication' not in win.store.project.analysis.studies
    # Export writes the kit in the background
    page.export(str(tmp_path))
    t = time.time()
    while not (tmp_path / 'paper_kit' / 'provenance.json').exists() and time.time() - t < 60:
        pump(app, 0.05)
    win.ctx.analysis.wait(10)
    pump(app, 0.2)
    assert (tmp_path / 'paper_kit' / 'main.tex').exists() and (tmp_path / 'export_manifest.json').exists()
    assert page.page_context()['title'] == 'Publication'
