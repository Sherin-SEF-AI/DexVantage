import json

from dexvantage import validation as VA


def test_every_check_passes_and_names_its_reference():
    res = VA.run()
    assert res.ok, [(c.id, c.error, c.exception) for c in res.failed]
    assert len(res.checks) >= 8
    for c in res.checks:
        assert c.reference and c.quantity and c.unit
        assert all(r in VA.REFERENCES for r in c.refs), c.id


def test_report_files(tmp_path):
    res = VA.run(settings=VA.ValidationSettings(quick=True))
    paths = VA.export(res, None, str(tmp_path))
    names = {p.split('/')[-1] for p in paths}
    assert {'validation.json', 'validation.csv', 'validation.html'} <= names
    doc = json.loads((tmp_path / 'validation.json').read_text())
    assert doc['schema'] == 'dexvantage.validation/1' and doc['summary']['ok']
    assert 'DexVantage validation report' in (tmp_path / 'validation.html').read_text()


def test_settings_are_tolerant_and_exit_codes():
    st = VA.ValidationSettings.from_dict({'models': 'frames,nonsense', 'seed': -5})
    assert st.models == ('frames',) and st.seed == 0
    res = VA.run(settings=st)
    assert {c.model for c in res.checks} == {'frames'} and VA.exit_code(res) == 0
    res.checks[0].passed = False
    assert VA.exit_code(res) == 1


def test_catalogue_lists_without_running():
    cat = VA.catalogue()
    assert {c['id'] for c in cat} >= {'frames.quat_to_matrix', 'formats.schema'}


def test_a_raising_check_fails_without_stopping_the_suite():
    from dexvantage.validation import suite
    spec = suite.Spec('x.boom', 'frames', 'Boom', 'q', 'r', 1.0, 'u', (), lambda ctx: 1 / 0)
    ctx = suite.Ctx(None, None, None, VA.ValidationSettings())
    c = suite.run_one(spec, ctx)
    assert not c.passed and 'ZeroDivisionError' in c.exception
