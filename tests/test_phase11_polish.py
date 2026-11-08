"""Phase 11 polish (open issues of docs/BUILD_STATE.md, each with its regression test): a change of a percentage in
percentage points; rig.run refusing a setting name its study does not read; the assistant's get_project_diff tool;
citation keys of a shared datasheet; the mlruns experiment meta.yaml keys of MLflow 3.17.2; plugin checks in
python -m dexvantage.validation; type hints on the public functions of the API modules (ruff's ANN rules); every skip
with its reason; every page's headline tooltip naming its model and inputs."""
import ast
import glob
import os
import re
import subprocess
import sys
import textwrap

import pytest

from dexvantage.core.project import Sensor, Variant, blank_project

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D400_URL = 'https://www.realsenseai.com/wp-content/uploads/2026/03/RealSense-D400-Series-Datasheet-Mar-2026.pdf'


def _env(tmp_path, extra_path=''):
    env = dict(os.environ)
    for k in ('XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_CACHE_HOME'):
        env[k] = str(tmp_path / ('env_' + k.lower()))
    env['PYTHONPATH'] = os.pathsep.join(p for p in (extra_path, ROOT, env.get('PYTHONPATH', '')) if p)
    env['QT_QPA_PLATFORM'] = 'offscreen'
    env.pop('DEXVANTAGE_NO_PLUGINS', None)
    return env


# ====================================================================== (1) percentage points
def test_a_change_of_a_percentage_is_in_percentage_points():
    from dexvantage.core import metrics as M
    assert M.delta_unit('%') == 'pp' and M.delta_unit(' % ') == 'pp'
    assert M.delta_unit('mm') == 'mm' and M.delta_unit('') == '' and M.delta_unit(None) == ''


def test_statistics_text_and_headline_write_pp():
    import numpy as np

    from dexvantage.core import stats as S
    from dexvantage.core.studies import stats as SS
    rng = np.random.default_rng(3)
    x = rng.normal(94.5, 0.2, 12)
    y = x - 0.18 + rng.normal(0, 0.01, 12)
    r = S.compare_samples(x, y, paired=True)
    v = S.verdict(r, r['p'], 'Task volume seen', '%')
    assert ' pp [' in v['text'] and not re.search(r'\d %\s*\[', v['text']), v['text']
    far = S.verdict(S.compare_samples(x, x + rng.normal(0, 0.3, 12), paired=True), 0.9, 'Seen', '%')
    assert 'pp (' in far['text'] or 'pp [' in far['text'], far['text']          # the detectable difference too
    assert ' mm [' in S.verdict(r, r['p'], 'Reach', 'mm')['text']               # other units unchanged
    doc = {'comparisons': [dict(r, id='a vs b', unit='%', p_adjusted=r['p'], verdict=v)],
           'correction': {'label': 'Holm', 'm': 1}}
    h = SS.headline(doc)
    assert h['unit'] == 'pp' and h['kind'] == 'samples'


def test_tracker_requirements_and_diff_deltas_write_pp():
    from dexvantage.core import projdiff as D
    from dexvantage.core import requirements as RQ
    from dexvantage.core import tracking as TR
    assert TR.delta_text(-0.18, '%') == '-0.18 pp' and TR.delta_text(0.5, 'mm') == '+0.5 mm'
    assert TR.delta_text(None, '%') == '—'
    card = {'variant': 'Rig A', 'errors': [], 'verdict_text': 'ready',
            'counts': {'pass': 1, 'fail': 0, 'not_evaluated': 0},
            'items': [{'name': 'Contact visibility', 'metric': 'grasp.contact_visibility', 'value': 100.0, 'unit': '%',
                       'bound': None, 'op_text': '≥', 'threshold': 95.0, 'is_default': True, 'margin': 5.0,
                       'status_text': 'pass', 'status': 'pass', 'reason': ''}]}
    t = RQ.table(card)
    assert '+5 pp' in t and '≥ 95 %' in t and '100 %' in t, t
    assert D._metric_text({'delta': -0.2014, 'unit': '%', 'verdict': 'worse'}) == '-0.2014 pp (worse)'
    assert D._metric_text({'delta': 3.0, 'unit': 'mm', 'verdict': 'better'}) == '+3 mm (better)'


# ====================================================================== (2) unknown setting names
def test_unknown_setting_names_are_refused_in_one_sentence():
    import dexvantage.api as dv
    from dexvantage.core import study_settings as SST
    from dexvantage.core.jobs import StudyError
    known = SST.known('tolerance')
    assert {'n', 'seed', 'scale', 'distribution', 'metrics', 'sensors'} <= set(known)
    rig = dv.new('Settings check')
    with pytest.raises(dv.DexVantageError) as e:
        rig.run('tolerance', draws=8)                       # the key is n: refused before anything runs
    msg = str(e.value)
    assert msg.count('.') == msg.count('. ') + 1 and msg.endswith('.')              # one sentence
    assert 'no setting called draws' in msg and 'n, ' in msg and 'seed' in msg
    with pytest.raises(dv.DexVantageError, match='no settings called cubeN and voxel'):
        rig.run('coverage', cubeN=128, voxel=5)
    with pytest.raises(dv.DexVantageError, match='no setting called draws'):
        dv.cluster.tolerance(rig, draws=8)
    SST.check('coverage', ['cube_n', 'q', 'sensors', 'voxel_mm'])                   # known names pass
    SST.check('audit', ['dataset', 'anything'])                                     # a study not listed: not checked
    assert SST.known('audit') is None
    with pytest.raises(StudyError):
        SST.check('bandwidth', ['episode'])


def test_every_listed_study_knows_its_defaults_and_its_settings_of_reads():
    from dexvantage.core import jobs
    from dexvantage.core import study_settings as SST
    jobs.load_studies()
    for sid in SST.EXTRA:
        assert sid in jobs.STUDIES, sid
        d = SST.defaults_of(sid)
        assert d, f'{sid} has no DEFAULTS: list its names explicitly'
        assert set(d) <= set(SST.known(sid)), sid


# ====================================================================== (3) get_project_diff
def _two_variant_project():
    p = blank_project('Diff cell')
    p.variants[0].name = 'Rig A'
    p.variants[0].sensors.append(Sensor(name='Wrist', model_id='intel-realsense-d405'))
    b = Variant.from_dict(p.variants[0].to_dict())
    b.uid, b.name = 'variantb01', 'Rig B'
    b.sensors.append(Sensor(name='Overhead', model_id='intel-realsense-d435'))
    p.variants.append(b)
    return p


def test_project_diff_tool_compares_a_variant_and_a_file(tmp_path):
    from dexvantage.assistant import tools as T
    from dexvantage.assistant.session import FileSession
    assert 'get_project_diff' in T.TOOLS and not T.TOOLS['get_project_diff'].changes
    p = _two_variant_project()
    s = FileSession(project=p)
    out = T.call(s, 'get_project_diff', {'variant': 'Rig B', 'metrics': ['rig.camera_count']})
    assert 'error' not in out, out
    assert out['compared'] == 'from the active variant Rig A to Rig B' and not out['identical']
    texts = ' '.join(c['text'] for c in out['changes'])
    assert 'Overhead' in texts and 'added' in texts and 'Wrist' not in texts, texts    # sensors paired by uid
    m = out['metrics'][0]
    assert m['id'] == 'rig.camera_count' and m['a'] == 1 and m['b'] == 2 and m['delta'] == 1, m
    path = tmp_path / 'before.dexv'
    old = p.snapshot()                                   # before: Rig A without its sensor, no Rig B
    old.variants = [old.variants[0]]
    old.variants[0].sensors = []
    old.save(str(path))
    out = T.call(s, 'get_project_diff', {'path': str(path)})
    assert 'error' not in out and out['compared'] == 'from before.dexv to this project', out
    assert any(c['section'] == 'sensors' for c in out['changes'])
    for args, words in (({}, 'Name another project file'), ({'variant': 'Rig A'}, 'is the active variant'),
                        ({'variant': 'Rig C'}, 'has no variant'), ({'path': str(tmp_path / 'none.dexv')},
                                                                     'There is no project at none.dexv'),
                        ({'path': str(path), 'variant': 'Rig B'}, 'not both'),
                        ({'variant': 'Rig B', 'metrics': ['no.such']}, 'There is no metric no.such')):
        err = T.call(s, 'get_project_diff', args)
        assert words in err.get('error', ''), (args, err)
        assert 'Traceback' not in err['error']
    (tmp_path / 'junk.dexv').write_text('not a zip', encoding='utf-8')
    err = T.call(s, 'get_project_diff', {'path': str(tmp_path / 'junk.dexv')})
    assert 'junk.dexv' in err.get('error', '') and 'Traceback' not in err['error'], err


# ====================================================================== (4) citation keys
def test_a_shared_datasheet_is_keyed_by_the_document_not_one_device():
    from dexvantage.core import cite as C
    from dexvantage.core.publication import CiteMap
    p = blank_project('UR5e-like rig')
    p.variants[0].sensors += [Sensor(name='Wrist D405', model_id='intel-realsense-d405'),
                              Sensor(name='Overhead D435', model_id='intel-realsense-d435')]
    lib = [e for e in C.entries(p, 'none') if e.group.startswith('Library sources')]
    shared = [e for e in lib if e.fields.get('url') == D400_URL]
    assert len(shared) == 1, [e.key for e in lib]
    e = shared[0]
    assert e.key == 'intel:realsense-d400-series-datasheet-mar-2026'
    assert 'intel-realsense-d405' in e.fields['note'] and 'intel-realsense-d435' in e.fields['note']
    assert not any(x.key in ('intel-realsense-d405', 'intel-realsense-d435') and x.fields.get('url') == D400_URL
                   for x in lib)
    keys = [x.key for x in C.entries(p, 'none')]
    assert len({k.lower() for k in keys}) == len(keys)
    cm = CiteMap(p)
    assert e.key in cm.library['intel-realsense-d405'] and e.key in cm.library['intel-realsense-d435']
    one = blank_project('D435 only')
    one.variants[0].sensors.append(Sensor(name='Overhead D435', model_id='intel-realsense-d435'))
    solo = next(x for x in C.entries(one, 'none') if x.fields.get('url') == D400_URL)
    assert solo.key.startswith('intel-realsense-d435')                    # one device: its own id
    assert C.shared_key('luxonis-oak-d', 'https://docs.luxonis.com/hardware/sensors/OV9282.md') == 'luxonis:ov9282'
    assert C.shared_key('acme-cam', 'Printed manual from the maker') == 'acme:printed-manual-from-the-maker'
    assert re.fullmatch(r'[A-Za-z][A-Za-z0-9_:.-]*', C.shared_key('x', 'https://e.org/a b%20c.pdf'))


# ====================================================================== (5) mlruns meta.yaml
def _yaml(path):
    import yaml
    with open(path, encoding='utf-8') as fh:
        return yaml.safe_load(fh)


def test_mlruns_experiment_meta_has_the_keys_mlflow_3_17_2_writes(tmp_path):
    from dexvantage.core.tracking import mlruns as ML
    ML._experiment(str(tmp_path), '123', 'DexVantage · cell', 1000)
    meta = _yaml(tmp_path / '123' / 'meta.yaml')
    assert set(meta) == {'artifact_location', 'creation_time', 'effective_trace_archival_retention', 'experiment_id',
                         'last_update_time', 'lifecycle_stage', 'name', 'trace_location', 'workspace'}
    assert meta['workspace'] == 'default' and meta['trace_location'] is None
    assert meta['effective_trace_archival_retention'] is None and meta['experiment_id'] == '123'


def test_mlruns_experiment_meta_matches_mlflow_own_file_store(tmp_path, monkeypatch):
    pytest.importorskip('mlflow', reason='mlflow-skinny (the tracking extra) is not installed')
    monkeypatch.setenv('MLFLOW_ALLOW_FILE_STORE', 'true')
    monkeypatch.setenv('MLFLOW_DISABLE_AGENT_HINT', '1')
    from mlflow.store.tracking.file_store import FileStore

    from dexvantage.core.tracking import mlruns as ML
    fs = FileStore(str(tmp_path / 'theirs'))
    eid = fs.create_experiment('demo')
    theirs = _yaml(tmp_path / 'theirs' / eid / 'meta.yaml')
    ML._experiment(str(tmp_path / 'ours'), '7', 'demo', 1000)
    ours = _yaml(tmp_path / 'ours' / '7' / 'meta.yaml')
    assert set(ours) == set(theirs)
    for k in ('effective_trace_archival_retention', 'lifecycle_stage', 'trace_location', 'workspace'):
        assert ours[k] == theirs[k], k
    exp = FileStore(str(tmp_path / 'ours')).get_experiment('7')         # MLflow reads ours back
    assert exp.name == 'demo' and exp.workspace == 'default'


# ====================================================================== (6) plugin checks in the validation CLI
PLUGIN = '''
def register(api):
    @api.check('polish.one_plus_one', 'One plus one against two', 'absolute error', 'the integer 2', 1e-12, '1')
    def _check(ctx):
        return api.Outcome(abs((1 + 1) - 2), {'sum': 2})
'''


def _install_plugin(site):
    di = site / 'polish_check_plugin-1.0.dist-info'
    di.mkdir(parents=True)
    (di / 'METADATA').write_text('Metadata-Version: 2.1\nName: polish-check-plugin\nVersion: 1.0\n', encoding='utf-8')
    (di / 'entry_points.txt').write_text('[dexvantage.plugins]\npolish_check = polish_check_plugin:register\n',
                                         encoding='utf-8')
    (di / 'RECORD').write_text('', encoding='utf-8')
    (site / 'polish_check_plugin.py').write_text(textwrap.dedent(PLUGIN), encoding='utf-8')


def test_validation_cli_runs_the_plugins_checks_unless_no_plugins(tmp_path):
    site = tmp_path / 'site'
    _install_plugin(site)
    env = _env(tmp_path, str(site))

    def cli(*args):
        return subprocess.run([sys.executable, '-m', 'dexvantage.validation', *args], cwd=str(tmp_path), env=env,
                              capture_output=True, text=True, timeout=300)
    on = cli('--list')
    assert on.returncode == 0, on.stderr[-2000:]
    assert 'polish.one_plus_one' in on.stdout
    off = cli('--list', '--no-plugins')
    assert off.returncode == 0 and 'polish.one_plus_one' not in off.stdout and 'frames.quat_to_matrix' in off.stdout
    run = cli(str(tmp_path / 'report'), '--models', 'plugin.polish_check')
    assert run.returncode == 0, run.stdout + run.stderr[-2000:]
    assert 'pass  polish.one_plus_one' in run.stdout and '1/1 checks passed' in run.stdout
    refused = cli(str(tmp_path / 'r2'), '--models', 'plugin.polish_check', '--no-plugins')
    assert refused.returncode == 4 and 'Unknown model id(s): plugin.polish_check' in refused.stderr
    assert 'Traceback' not in refused.stderr


# ====================================================================== (7) type hints on public functions
ANN_FILES = ('dexvantage/api.py', 'dexvantage/core/metrics.py', 'dexvantage/core/factors.py',
             'dexvantage/core/exporters/__init__.py', 'dexvantage/core/stats.py',
             'dexvantage/core/tracking/__init__.py')


def test_api_modules_annotate_every_parameter_and_public_return():
    """The ANN rules pyproject.toml selects for these modules (ANN001-003, ANN201, ANN206), checked by an AST walk
    written here (independent of ruff)."""
    bad = []
    for rel in ANN_FILES:
        tree = ast.parse(open(os.path.join(ROOT, rel), encoding='utf-8').read())
        for n in ast.walk(tree):
            if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            a = n.args
            for x in a.posonlyargs + a.args + a.kwonlyargs + [y for y in (a.vararg, a.kwarg) if y]:
                if x.annotation is None and x.arg not in ('self', 'cls'):
                    bad.append(f'{rel}:{n.lineno} {n.name}({x.arg})')
            if not n.name.startswith('_') and n.returns is None:
                bad.append(f'{rel}:{n.lineno} {n.name} returns')
    assert not bad, bad


def test_ruff_selects_the_annotation_rules_for_those_modules_only():
    tomllib = pytest.importorskip('tomllib', reason='tomllib needs Python 3.11 or newer')
    with open(os.path.join(ROOT, 'pyproject.toml'), 'rb') as fh:
        lint = tomllib.load(fh)['tool']['ruff']['lint']
    assert {'ANN001', 'ANN002', 'ANN003', 'ANN201', 'ANN206'} <= set(lint['select'])
    neg = [k for k, v in lint['per-file-ignores'].items() if k.startswith('!') and v == ['ANN']]
    assert len(neg) == 1 and all(f.split('dexvantage/', 1)[1] in neg[0] for f in ANN_FILES)


# ====================================================================== (8) every skip states its reason
def _skip_calls(tree):
    funcs = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = ast.unparse(n.func)
            if f in ('pytest.skip', 'pytest.xfail', 'pytest.importorskip', 'pytest.mark.skipif', 'pytest.mark.skip',
                     'pytest.mark.xfail'):
                yield f, n
        elif (isinstance(n, ast.Attribute) and id(n) not in funcs
              and ast.unparse(n) in ('pytest.mark.skip', 'pytest.mark.xfail')):
            yield ast.unparse(n), None                                   # a bare decorator: no reason at all


def _has_reason(f, call):
    if call is None:
        return False
    kw = {k.arg: k.value for k in call.keywords}
    if f in ('pytest.skip', 'pytest.xfail'):
        v = call.args[0] if call.args else kw.get('reason')
    elif f == 'pytest.mark.skip':
        v = kw.get('reason') or (call.args[0] if call.args else None)
    else:
        v = kw.get('reason')
    if v is None:
        return False
    if isinstance(v, ast.Constant):
        return isinstance(v.value, str) and len(v.value.strip()) > 8
    return True                                                          # a sentence built at run time (f-string, why)


def test_every_skip_and_xfail_states_its_reason():
    bad, seen = [], 0
    for path in sorted(glob.glob(os.path.join(ROOT, 'tests', '**', '*.py'), recursive=True) +
                       glob.glob(os.path.join(ROOT, 'examples', '**', '*.py'), recursive=True)):
        tree = ast.parse(open(path, encoding='utf-8').read())
        for f, call in _skip_calls(tree):
            seen += 1
            if not _has_reason(f, call):
                bad.append(f'{os.path.relpath(path, ROOT)}:{getattr(call, "lineno", "?")} {f}')
    assert seen > 60 and not bad, bad


# ====================================================================== (9) headline tooltips: model and inputs
def test_every_page_headline_tooltip_names_its_model_and_inputs(win):
    from dexvantage.core import jobs
    from dexvantage.ui.pages import PAGES, load_pages
    load_pages()
    jobs.load_studies()
    assert len(PAGES) >= 30
    for pid, info in PAGES.items():
        page = win.pages[pid]
        tip = page.head.figure.toolTip()
        assert 'Model: ' in tip and 'Inputs: ' in tip, (pid, tip)
        if info.cls.STUDY and not info.cls.MODEL:
            mod = jobs.STUDIES[info.cls.STUDY].fn.__module__.split('.', 1)[1].replace('.', '/') + '.py'
            assert mod in tip and jobs.STUDIES[info.cls.STUDY].title in tip, (pid, tip)
        elif not info.cls.STUDY:
            assert info.cls.MODEL, f'{pid} runs no study: give it a MODEL sentence'
    cov = win.pages['coverage']
    cov.doc = {'settings': {'cube_n': 512, 'good_pct': 80.0, 'backend': 'auto', 'sensors': [],
                            'defaults': ['good_pct']}}
    cov.head.set('69.2', '%', 'good', 'well seen', 'text', 'Share of the task volume seen by two or more cameras.')
    tip = cov.head.figure.toolTip()
    assert tip.startswith('Share of the task volume seen by two or more cameras.')
    assert 'core/studies/coverage.py' in tip and 'cube_n 512' in tip and 'good_pct 80 (default)' in tip, tip
    assert 'sensors' not in tip and 'the rig variant Rig A as it is now' in tip, tip
