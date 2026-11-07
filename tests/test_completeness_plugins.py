"""A plugin's validation checks (plugin_sdk PluginAPI.check; validation/suite.py PLUGIN_MODELS): docs/DEVELOPING.md
asks every model to be validated against an independent reference, and a plugin's study is a model too. The check
joins its own model ('plugin.<name>'), runs with the built-in checks, cites the plugin's references in the report and
in BibTeX, and goes again when the plugin is unloaded or fails."""
import importlib
import json
import sys
import textwrap
from importlib.metadata import EntryPoint

import pytest

PLUGIN = '''
    import numpy as np

    def register(api):
        @api.study('toy_mean', 'Toy mean', quick=True)
        def toy(project, ctx):
            return {'summary': {'mean': float(sum([1.0, 2.0, 4.0]) / 3)}}

        @api.check('toy.mean_vs_numpy', 'Toy mean against NumPy', 'absolute error of the mean', 'numpy.mean',
                   1e-12, '1', references={'toy_numpy2020': 'C. R. Harris et al., "Array programming with NumPy", '
                                                            'Nature 585, 357-362 (2020), https://doi.org/10.1038/'
                                                            's41586-020-2649-2 (read 2026-11-07).'},
                   model_title='Toy mean (plugin)')
        def _check(ctx):
            x = [1.0, 2.0, 4.0]
            ours = toy(None, None)['summary']['mean']
            return api.Outcome(abs(ours - float(np.mean(x))), {'ours': ours, 'numpy': float(np.mean(x))})
'''


@pytest.fixture
def plugin(tmp_path, monkeypatch):
    from dexvantage import plugin_sdk as PL
    from dexvantage.core import jobs
    from dexvantage.validation import references as VR
    from dexvantage.validation import suite as VS
    VS.load_checks()
    PL._builtins_loaded()
    saved = (list(PL._RECORDS), PL._LOADED, PL._ENABLED, dict(PL.ORIGIN), dict(jobs.STUDIES), list(VS.CHECKS),
             list(VS.PLUGIN_MODELS), dict(VR.REFERENCES))
    name = 'toy_check_plugin'

    def load(body=PLUGIN):
        (tmp_path / f'{name}.py').write_text(textwrap.dedent(body), encoding='utf-8')
        monkeypatch.syspath_prepend(str(tmp_path))
        importlib.invalidate_caches()
        sys.modules.pop(name, None)
        return PL.load(entry_points=[EntryPoint(name=name, value=f'{name}:register', group='dexvantage.plugins')])[0]
    yield load
    for r in list(PL._RECORDS):
        PL.unload(r.name)
    PL._RECORDS[:] = saved[0]
    PL._LOADED, PL._ENABLED = saved[1], saved[2]
    PL.ORIGIN.clear()
    PL.ORIGIN.update(saved[3])
    jobs.STUDIES.clear()
    jobs.STUDIES.update(saved[4])
    VS.CHECKS[:] = saved[5]
    VS.PLUGIN_MODELS[:] = saved[6]
    VR.REFERENCES.clear()
    VR.REFERENCES.update(saved[7])
    sys.modules.pop(name, None)


def test_plugin_check_runs_in_the_suite_and_is_reported(plugin, tmp_path):
    from dexvantage import plugin_sdk as PL
    from dexvantage import validation as VA
    from dexvantage.validation import suite as VS
    r = plugin()
    assert r.status == 'loaded', r.error
    assert r.added['check'] == ['toy.mean_vs_numpy'] and r.added['study'] == ['toy_mean']
    assert PL.summary_text(r) == '1 study (toy_mean), 1 validation check (toy.mean_vs_numpy)'
    assert PL.origin('check', 'toy.mean_vs_numpy') == 'toy_check_plugin'
    assert 'plugin.toy_check_plugin' in VA.available_models() and VS.MODEL_IDS[-1] != 'plugin.toy_check_plugin'
    res = VA.run(settings={'models': ['plugin.toy_check_plugin']})
    (c,) = res.checks
    assert c.id == 'toy.mean_vs_numpy' and c.passed and c.error <= 1e-12
    (m,) = res.by_model()
    assert m[0] == 'plugin.toy_check_plugin' and m[1] == 'Toy mean (plugin)'
    doc = VA.to_doc(res)
    assert doc['models'][0]['id'] == 'plugin.toy_check_plugin'
    assert 'toy_numpy2020' in json.dumps(doc)
    html = dict(VA.files(res))['validation.html'].decode('utf-8')
    assert 'Array programming with NumPy' in html and 'Toy mean (plugin)' in html
    # the BibTeX of the checks' references includes it
    from dexvantage.core import cite as C
    assert 'toy_numpy2020' in C.bibtex(None, 'checks')


def test_unload_removes_check_model_and_reference(plugin):
    from dexvantage import plugin_sdk as PL
    from dexvantage import validation as VA
    from dexvantage.validation import references as VR
    from dexvantage.validation import suite as VS
    n = len(VS.CHECKS)
    plugin()
    assert len(VS.CHECKS) == n + 1 and 'toy_numpy2020' in VR.REFERENCES
    PL.unload('toy_check_plugin')
    assert len(VS.CHECKS) == n and 'toy_numpy2020' not in VR.REFERENCES
    assert 'plugin.toy_check_plugin' not in VA.available_models() and not VS.PLUGIN_MODELS


def test_bad_checks_are_skipped_with_a_sentence(plugin):
    from dexvantage.validation import suite as VS
    r = plugin('''
        def register(api):
            api.check('coverage.no_such_but_new', 'x', 'q', 'r', 0.1, '1', refs=['no_such_key'])(lambda ctx: None)
            api.check('dupe', 'x', 'q', 'r', 0.1, '1', references={'arvo1990': 'other text'})(lambda ctx: None)
            api.check('ok.one', 'x', 'q', 'r', 0.1, '1')(lambda ctx: api.Outcome(0.0))
            api.check('ok.one', 'x', 'q', 'r', 0.1, '1')(lambda ctx: api.Outcome(0.0))
    ''')
    assert r.status == 'loaded' and r.added['check'] == ['ok.one']
    assert len(r.problems) == 3
    assert 'reference key no_such_key, which neither DexVantage nor the plugin defines' in r.problems[0]
    assert 'reference key arvo1990, which another source already has with other text' in r.problems[1]
    assert 'an id another check already has' in r.problems[2]
    assert sum(1 for c in VS.CHECKS if c.id == 'ok.one') == 1


def test_a_failing_plugin_takes_its_checks_away(plugin):
    from dexvantage.validation import suite as VS
    n = len(VS.CHECKS)
    r = plugin('''
        def register(api):
            api.check('gone.soon', 'x', 'q', 'r', 0.1, '1', references={'gone_ref': 'G. One (read 2026-11-07).'})(
                lambda ctx: api.Outcome(0.0))
            raise KeyError('boom')
    ''')
    assert r.status == 'error' and 'What it had added before the failure was removed.' in r.error
    assert len(VS.CHECKS) == n and not VS.PLUGIN_MODELS


def test_batch_models_flag_knows_plugin_models():
    import inspect

    from dexvantage import batch
    src = inspect.getsource(batch)
    assert 'VA.available_models()' in src and 'not in VA.MODEL_IDS' not in src


def test_a_builtin_page_over_a_plugin_page_is_said(plugin):
    """The window loads its pages after the plugins: a plugin page with a built-in page's id is replaced, and the
    plugin's record says so (it was silent before)."""
    from dexvantage import plugin_sdk as PL
    from dexvantage.ui import pages as PG
    PG.load_pages()
    saved = dict(PG.PAGES)
    try:
        PG.PAGES.pop('coverage', None)            # as at start-up, before the window imports its page modules
        r = plugin('''
            def register(api):
                api.page('coverage', 'My coverage', 'nowhere:Page')
        ''')
        assert r.added['page'] == ['coverage'] and PL.origin('page', 'coverage') == 'toy_check_plugin'
        PG.register('coverage', 'Coverage and resolution', saved['coverage'].cls)   # the built-in module's register
        assert r.added['page'] == [] and PL.origin('page', 'coverage') is None
        assert r.problems == ['toy_check_plugin tried to add the page "coverage", an id a DexVantage page has; the '
                              'built-in page is shown instead.']
        PG.register('coverage', 'Coverage and resolution', saved['coverage'].cls)   # built-in again: nothing to say
        assert len(r.problems) == 1
    finally:
        PG.PAGES.clear()
        PG.PAGES.update(saved)
