"""The plugin SDK (dexvantage/plugin_sdk.py): the template (examples/plugin-template) installed into a temporary target
folder and found through importlib.metadata as an installed distribution, or loaded through a fake entry point; what a
plugin may add (studies with their batch flag, metrics, exporters, dataset readers, library folders, pages); failures
explained in a sentence and rolled back without breaking anything; --no-plugins; Help ▸ Plugins."""
import importlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
from importlib.metadata import EntryPoint

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
TEMPLATE = os.path.join(ROOT, 'examples', 'plugin-template')
PKG = 'dexvantage_plugin_template'


# ====================================================================== helpers
def _template_entry_points() -> dict:
    """The template's declared entry points ({name: value}), read from its pyproject.toml."""
    tomllib = pytest.importorskip('tomllib')
    with open(os.path.join(TEMPLATE, 'pyproject.toml'), 'rb') as fh:
        doc = tomllib.load(fh)
    return doc['project']['entry-points']['dexvantage.plugins'], doc['project']['name'], doc['project']['version']


def install_template(target: str) -> str:
    """Install the template into target the way pip --target lays it out (the package and a .dist-info folder with
    METADATA and entry_points.txt), without a build backend or the network."""
    eps, name, version = _template_entry_points()
    shutil.copytree(os.path.join(TEMPLATE, PKG), os.path.join(target, PKG),
                    ignore=shutil.ignore_patterns('__pycache__'))
    di = os.path.join(target, f'{name.replace("-", "_")}-{version}.dist-info')
    os.makedirs(di)
    with open(os.path.join(di, 'METADATA'), 'w', encoding='utf-8') as fh:
        fh.write(f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n')
    with open(os.path.join(di, 'entry_points.txt'), 'w', encoding='utf-8') as fh:
        fh.write('[dexvantage.plugins]\n' + ''.join(f'{k} = {v}\n' for k, v in eps.items()))
    with open(os.path.join(di, 'RECORD'), 'w', encoding='utf-8') as fh:
        fh.write('')
    return target


def _forget_template():
    for m in [m for m in sys.modules if m == PKG or m.startswith(PKG + '.')]:
        del sys.modules[m]


@pytest.fixture
def registries():
    """Every registry a plugin can touch, restored after the test (the rest of the suite sees the built-ins only)."""
    from dexvantage import plugin_sdk as PL
    from dexvantage.core import exporters as EX
    from dexvantage.core import jobs
    from dexvantage.core import library as LB
    from dexvantage.core import metrics as M
    from dexvantage.core.datasets import episode as EP
    from dexvantage.ui import pages as PG
    PL._builtins_loaded()
    PG.load_pages()             # the built-in pages first: restoring a snapshot taken before they loaded emptied them
    saved = (dict(jobs.STUDIES), dict(M.METRICS), dict(EX.EXPORTERS), list(EP.READERS), dict(EP.FORMAT_TITLES),
             list(LB.EXTRA_DIRS), dict(PG.PAGES), list(PL._RECORDS), PL._LOADED, PL._ENABLED, dict(PL.ORIGIN))
    yield PL
    for r in list(PL._RECORDS):
        PL.unload(r.name)
    for reg, old in ((jobs.STUDIES, saved[0]), (M.METRICS, saved[1]), (EX.EXPORTERS, saved[2]),
                     (EP.FORMAT_TITLES, saved[4]), (PG.PAGES, saved[6]), (PL.ORIGIN, saved[10])):
        reg.clear()
        reg.update(old)
    EP.READERS[:] = saved[3]
    PL._RECORDS[:] = saved[7]
    PL._LOADED, PL._ENABLED = saved[8], saved[9]
    if LB.EXTRA_DIRS != saved[5]:
        LB.EXTRA_DIRS[:] = saved[5]
        LB.reload()
    _forget_template()


def _module_plugin(tmp_path, monkeypatch, name: str, body: str) -> EntryPoint:
    """A plugin module written to tmp_path, on sys.path, with a fake entry point to its register."""
    (tmp_path / f'{name}.py').write_text(textwrap.dedent(body), encoding='utf-8')
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    sys.modules.pop(name, None)
    return EntryPoint(name=name, value=f'{name}:register', group='dexvantage.plugins')


def _env(tmp_path, extra_path=''):
    env = dict(os.environ)
    for k in ('XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_CACHE_HOME'):
        env[k] = str(tmp_path / ('env_' + k.lower()))
    env['PYTHONPATH'] = os.pathsep.join(p for p in (extra_path, ROOT, env.get('PYTHONPATH', '')) if p)
    env['QT_QPA_PLATFORM'] = 'offscreen'
    env.pop('DEXVANTAGE_NO_PLUGINS', None)
    return env


# ====================================================================== discovery and loading
def test_installed_template_is_found_and_loaded(tmp_path, monkeypatch, registries):
    PL = registries
    target = install_template(str(tmp_path / 'site'))
    eps = PL.discover(paths=[target])
    assert [(e.name, e.value) for e in eps] == [('aim_template', f'{PKG}:register')]
    monkeypatch.syspath_prepend(target)
    recs = PL.load(paths=[target])
    assert len(recs) == 1
    r = recs[0]
    assert r.status == 'loaded', r.error
    assert (r.distribution, r.version) == ('dexvantage-plugin-template', '0.1.0')
    assert r.added['study'] == ['aim_check'] and r.added['metric'] == ['aim.worst_deg', 'aim.share_aimed']
    assert r.added['page'] == ['aim_check']
    assert PL.summary_text(r) == ('1 study (aim_check), 2 metrics (aim.worst_deg, aim.share_aimed), '
                                  '1 page (aim_check)')
    assert PL.origin('study', 'aim_check') == 'aim_template'
    from dexvantage.ui import pages as PG
    assert PG.PAGES['aim_check'].title == 'Aim at the task volume'
    assert any('aim_template 0.1.0: loaded; added 1 study (aim_check)' in line for line in PL.report_lines())


def test_plugin_study_through_the_api_metrics_and_requirements(tmp_path, monkeypatch, registries):
    PL = registries
    target = install_template(str(tmp_path / 'site'))
    monkeypatch.syspath_prepend(target)
    PL.load(paths=[target])
    import dexvantage as dv
    st = {r['id']: r for r in dv.studies()}
    assert st['aim_check']['plugin'] == 'aim_template' and st['aim_check']['batch'] == '--aim-check'
    assert st['coverage']['plugin'] is None
    assert next(m for m in dv.metrics() if m['id'] == 'aim.worst_deg')['plugin'] == 'aim_template'
    rig = dv.sample('ur5e')
    res = rig.run('aim-check')
    assert res.study == 'aim_check' and res.headline['metric'] == 'aim.worst_deg'
    assert res.headline['value'] == pytest.approx(res['summary']['worst_deg'])
    m = rig.metrics(['aim.worst_deg', 'aim.share_aimed'])
    assert m['aim.share_aimed']['value'] == pytest.approx(res['summary']['share_aimed_pct'])
    card = rig.check([{'metric': 'aim.worst_deg', 'op': '<=', 'threshold': 180}])
    assert card['verdict'] == 'pass'
    import argparse
    from dexvantage import batch
    ap = argparse.ArgumentParser()
    batch.add_arguments(ap)
    assert '--aim-check' in ap.format_help()


def test_fake_entry_point(registries):
    PL = registries
    sys.path.insert(0, TEMPLATE)
    try:
        ep = EntryPoint(name='aim_template', value=f'{PKG}:register', group=PL.GROUP)
        r = PL.load(entry_points=[ep])[0]
    finally:
        sys.path.remove(TEMPLATE)
    assert r.status == 'loaded' and r.version == '0.1.0'          # no distribution: the module's __version__
    assert r.distribution == ''
    PL.unload('aim_template')
    from dexvantage.core import jobs, metrics
    assert 'aim_check' not in jobs.STUDIES and 'aim.worst_deg' not in metrics.METRICS
    assert PL.summary_text(r) == 'nothing'


def test_a_failing_plugin_is_rolled_back_and_explained(tmp_path, monkeypatch, registries):
    PL = registries
    ep = _module_plugin(tmp_path, monkeypatch, 'broken_plugin', '''
        def register(api):
            api.study('broken_study', 'Broken')(lambda project, ctx: {})
            raise RuntimeError('the config file is missing')
    ''')
    missing = EntryPoint(name='missing_plugin', value='no_such_module_dv:register', group=PL.GROUP)
    newer = _module_plugin(tmp_path, monkeypatch, 'newer_plugin', '''
        REQUIRES_API = 99
        def register(api):
            api.study('newer_study', 'Newer')(lambda project, ctx: {})
    ''')
    recs = {r.name: r for r in PL.load(entry_points=[ep, missing, newer])}
    from dexvantage.core import jobs
    b = recs['broken_plugin']
    assert b.status == 'error' and 'broken_study' not in jobs.STUDIES
    assert b.error == ('broken_plugin could not be loaded: RuntimeError: the config file is missing. What it had '
                       'added before the failure was removed.')
    assert 'Traceback' not in b.error
    m = recs['missing_plugin']
    assert m.status == 'error' and 'could not be imported' in m.error and 'pip uninstall missing_plugin' in m.error
    n = recs['newer_plugin']
    assert n.status == 'error' and 'needs version 99 of the plugin interface' in n.error
    assert 'newer_study' not in jobs.STUDIES


def test_duplicate_ids_are_skipped_not_replaced(tmp_path, monkeypatch, registries):
    PL = registries
    from dexvantage.core import exporters as EX
    from dexvantage.core import jobs
    from dexvantage.core import metrics as M
    cov = jobs.STUDIES['coverage'].fn
    ep = _module_plugin(tmp_path, monkeypatch, 'dupe_plugin', '''
        def register(api):
            api.study('coverage', 'Not coverage')(lambda project, ctx: {})
            api.study('own_study', 'Own', batch='coverage')(lambda project, ctx: {})
            api.metric('coverage.covered', 'x', '%', 'higher', study='coverage')(lambda doc: 0)
            api.exporter('mjcf', 'x')(lambda *a: {})
            api.study('fine_study', 'Fine')(lambda project, ctx: {'summary': {}})
    ''')
    r = PL.load(entry_points=[ep])[0]
    assert r.status == 'loaded' and r.added['study'] == ['fine_study']
    assert jobs.STUDIES['coverage'].fn is cov and M.METRICS['coverage.covered'].study == 'coverage'
    assert EX.EXPORTERS['mjcf'].title != 'x'
    assert len(r.problems) == 4
    assert r.problems[0] == ('dupe_plugin tried to add the study "coverage", which DexVantage already has; it was not '
                             'added.')
    assert 'batch flag --coverage' in r.problems[1]


def test_bad_study_id_is_an_error(tmp_path, monkeypatch, registries):
    PL = registries
    ep = _module_plugin(tmp_path, monkeypatch, 'dotted_plugin', '''
        def register(api):
            api.study('my.study', 'Dotted')(lambda project, ctx: {})
    ''')
    r = PL.load(entry_points=[ep])[0]
    assert r.status == 'error' and 'is not lower-case words joined by underscores' in r.error


def test_exporter_reader_and_library_folder(tmp_path, monkeypatch, registries):
    PL = registries
    from dexvantage.core import library as LB
    builtin = LB.entries('camera')[0]
    lib = tmp_path / 'plugin_lib' / 'camera'
    lib.mkdir(parents=True)
    d = json.loads(json.dumps(builtin.data))
    d['id'] = 'plugin-test-camera'
    d['name'] = 'Plugin test camera'
    (lib / 'plugin-test-camera.json').write_text(json.dumps(d), encoding='utf-8')
    ep = _module_plugin(tmp_path, monkeypatch, 'kit_plugin', f'''
        import json, os
        import numpy as np
        LIB = {str(tmp_path / 'plugin_lib')!r}

        def register(api):
            from dexvantage.core.datasets.episode import Dataset, Episode, EpisodeRef

            @api.exporter('toy_format', 'Toy format')
            def _export(project, out_dir, options, ctx):
                with open(os.path.join(out_dir, 'toy.json'), 'w') as fh:
                    json.dump({{'sensors': len(project.variant.sensors)}}, fh)
                return {{'files': ['toy.json'], 'notes': ['A toy.']}}

            class Toy(Dataset):
                format, version = 'toy', '1'
                def __init__(self, path):
                    self.path, self.info = path, {{}}
                def episodes(self):
                    return [EpisodeRef(0, 'only', 3)]
                def load(self, i):
                    return Episode('only', 0, np.arange(3) / 10, np.zeros((3, 1)), ['j'], fps=10.0)

            api.dataset_reader('toy', sniff=lambda p: os.path.isfile(os.path.join(p, 'toy.marker')),
                               title='Toy recordings')(Toy)
            api.library_folder(LIB)
    ''')
    r = PL.load(entry_points=[ep])[0]
    assert r.status == 'loaded', r.error
    assert PL.summary_text(r) == ('1 exporter (toy_format), 1 dataset reader (toy), 1 library folder (plugin_lib)')
    e = LB.get('plugin-test-camera')
    assert e is not None and e.origin == 'the plugin kit_plugin' and not e.is_custom
    import dexvantage as dv
    rig = dv.sample('so100')
    man = rig.export(str(tmp_path / 'out'), ['toy_format'])
    assert man['formats']['toy_format']['status'] == 'ok' and 'toy.json' in man['files']
    rec_dir = tmp_path / 'rec'
    rec_dir.mkdir()
    (rec_dir / 'toy.marker').write_text('', encoding='utf-8')
    with dv.datasets.open(str(rec_dir)) as ds:
        assert ds.format == 'toy' and len(ds.episodes()) == 1 and len(ds.episode_rows(0)) == 3
        assert 'Toy recordings 1' in repr(ds)
    assert next(f for f in dv.datasets.formats() if f['format'] == 'toy')['plugin'] == 'kit_plugin'
    PL.unload('kit_plugin')
    assert LB.get('plugin-test-camera') is None
    with pytest.raises(dv.DexVantageError):
        dv.datasets.open(str(rec_dir))


def test_off_finds_without_loading(tmp_path, monkeypatch, registries):
    PL = registries
    target = install_template(str(tmp_path / 'site'))
    monkeypatch.syspath_prepend(target)
    r = PL.load(enabled=False, paths=[target])[0]
    assert r.status == 'off' and not PL.enabled() and PL.summary_text(r) == 'nothing (plugins are off)'
    from dexvantage.core import jobs
    assert 'aim_check' not in jobs.STUDIES
    monkeypatch.setenv('DEXVANTAGE_NO_PLUGINS', '1')
    assert PL.load(paths=[target], reload=True)[0].status == 'off'


# ====================================================================== start-up, the batch flag, the template's tests
def test_startup_loads_installed_plugins_and_no_plugins_flag(tmp_path):
    target = install_template(str(tmp_path / 'site'))
    env = _env(tmp_path, target)
    on = subprocess.run([sys.executable, '-m', 'dexvantage', '--list-plugins'], cwd=str(tmp_path), env=env,
                        capture_output=True, text=True, timeout=300)
    assert on.returncode == 0, on.stderr[-2000:]
    assert 'aim_template 0.1.0: loaded; added 1 study (aim_check), 2 metrics' in on.stdout
    off = subprocess.run([sys.executable, '-m', 'dexvantage', '--no-plugins', '--list-plugins'], cwd=str(tmp_path),
                         env=env, capture_output=True, text=True, timeout=300)
    assert off.returncode == 0 and 'aim_template 0.1.0: off' in off.stdout, off.stdout + off.stderr[-2000:]
    out = tmp_path / 'aim.json'
    run = subprocess.run([sys.executable, '-m', 'dexvantage', '--sample', 'ur5e', '--aim-check', str(out)],
                         cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stdout + run.stderr[-2000:]
    doc = json.loads(out.read_text(encoding='utf-8'))
    assert doc['study'] == 'aim_check' and doc['summary']['total'] == 2 and 'provenance' in doc
    refused = subprocess.run([sys.executable, '-m', 'dexvantage', '--sample', 'ur5e', '--no-plugins', '--aim-check',
                              str(out)], cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=300)
    assert refused.returncode == 2 and 'Traceback' not in refused.stderr


def test_a_broken_installed_plugin_does_not_stop_the_app(tmp_path):
    site = tmp_path / 'site'
    di = site / 'broken_dv_plugin-1.0.dist-info'
    di.mkdir(parents=True)
    (di / 'METADATA').write_text('Metadata-Version: 2.1\nName: broken-dv-plugin\nVersion: 1.0\n', encoding='utf-8')
    (di / 'entry_points.txt').write_text('[dexvantage.plugins]\nbroken = broken_dv_plugin:register\n',
                                         encoding='utf-8')
    (site / 'broken_dv_plugin.py').write_text('def register(api):\n    raise KeyError("sensors")\n', encoding='utf-8')
    out = subprocess.run([sys.executable, '-m', 'dexvantage', '--list-plugins'], cwd=str(tmp_path),
                         env=_env(tmp_path, str(site)), capture_output=True, text=True, timeout=300)
    assert out.returncode == 0
    assert "broken 1.0: could not be loaded: KeyError: 'sensors'." in out.stdout


def test_the_template_own_tests_pass(tmp_path):
    env = _env(tmp_path, TEMPLATE)
    out = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests'], cwd=TEMPLATE,
                         env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-2000:]
    assert '4 passed' in out.stdout


def test_pip_install_of_the_template(tmp_path, registries):
    """A real pip install --target (no network, no build isolation): runs only where setuptools is installed."""
    pytest.importorskip('setuptools')
    src = tmp_path / 'src'
    shutil.copytree(TEMPLATE, src, ignore=shutil.ignore_patterns('__pycache__'))
    target = tmp_path / 'site'
    out = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '--no-index',
                          '--target', str(target), str(src)], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert [e.name for e in registries.discover(paths=[str(target)])] == ['aim_template']


# ====================================================================== the window: the page and Help ▸ Plugins
def test_window_page_and_help_plugins(app, settings, tmp_path, monkeypatch, registries):
    PL = registries
    target = install_template(str(tmp_path / 'site'))
    monkeypatch.syspath_prepend(target)
    broken = _module_plugin(tmp_path, monkeypatch, 'broken_ui_plugin', '''
        def register(api):
            raise ValueError('no licence file')
    ''')
    eps = PL.discover(paths=[target]) + [broken]
    PL.load(entry_points=eps)
    from dexvantage.core.robot.samples import sample_project
    from dexvantage.ui.dialogs import plugins as DP
    from dexvantage.ui.mainwindow import MainWindow
    from conftest import pump
    from dexvantage.core import factors as F
    project = F.apply(sample_project('ur5e'), {'sensor:Overhead D435.dyaw': 60.0})   # turned away: a suggestion
    w = MainWindow(project, soft_3d=True, settings=settings)
    try:
        w.show()
        assert 'aim_check' in w.pages and 'page_aim_check' in w.actions_by_id
        assert 'plugins' in w.actions_by_id and w.actions_by_id['plugins'] in w.menus['Help'].actions()
        page = w.pages['aim_check']
        from dexvantage.core import jobs
        doc = jobs.run_study('aim_check', w.store.project.snapshot())
        page._finished('aim_check', doc)
        assert page.head.state['word'] == doc['summary']['word']
        assert page.table.topLevelItemCount() == 2
        assert len(page.overlays(w.viewport)) == 2
        sg = doc['suggestion']
        assert sg and doc['summary']['word'] == 'aimed away' and page.b_apply.isEnabled()
        page.preview()
        assert any(d.dash for d in page.overlays(w.viewport))
        from dexvantage.core import camrender as CR
        from dexvantage.core import library as LB
        from dexvantage.core.studies import coverage as CV
        s = next(x for x in w.store.project.variant.sensors if x.name == 'Wrist D405')
        im, _w = CV.pick_imager(s, LB.sensor_imagers(s, CV.sensor_mode(s, {})), {})
        ov = page.camview_overlays(im.camera, CR.optical_pose(w.store.project, s, im, None))
        assert [o[2] for o in ov] == ['image centre', 'task volume centre']      # the wrist D405 sees the centre
        page.apply()
        after = jobs.run_study('aim_check', w.store.project.snapshot())
        row = next(r for r in after['cameras'] if r['sensor_uid'] == sg['sensor_uid'])
        assert row['angle_deg'] < 1e-6
        w.store.undo.undo()                                      # one undo step restores the mount
        back = jobs.run_study('aim_check', w.store.project.snapshot())
        assert back['summary'] == doc['summary']
        d = DP.open_dialog(w, modal=False)
        pump(app, 0.05)
        text = d.text()
        assert '1 of 2 installed plugins loaded' in text
        assert 'aim_template 0.1.0 (dexvantage-plugin-template) · loaded. Added: 1 study (aim_check)' in text
        assert 'broken_ui_plugin could not be loaded: ValueError: no licence file.' in text
        assert 'Traceback' not in text
        d.close()
        PL._ENABLED = False
        assert 'Plugins are off for this session' in DP.intro_text()
    finally:
        w.cancel_jobs()
        w.hide()
        w.deleteLater()
        pump(app, 0.05)
