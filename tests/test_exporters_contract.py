"""The exporter registry contract (core/exporters/__init__.py)."""
import json
import os

from dexvantage.core import exporters as X
from dexvantage.core.jobs import StudyError
from dexvantage.core.project import Project


def test_export_writes_a_manifest_and_records_refusals(tmp_path):
    @X.exporter('_t_ok', 'Test format')
    def _ok(project, out, options, ctx):
        with open(os.path.join(out, 'a.json'), 'w') as fh:
            json.dump({'x': options.get('x', 1)}, fh)
        return {'files': ['a.json'], 'notes': ['a note']}

    @X.exporter('_t_no', 'Refusing format')
    def _no(project, out, options, ctx):
        raise StudyError('It needs a policy.')

    try:
        man = X.export(Project(), str(tmp_path), ['_t_ok', '_t_no'], options={'_t_ok': {'x': 2}})
        assert man['formats']['_t_ok']['status'] == 'ok' and man['formats']['_t_no']['status'] == 'refused'
        assert 'policy' in man['formats']['_t_no']['reason'] and 'a.json' in man['files']
        doc = json.load(open(tmp_path / 'a.json'))
        assert doc['x'] == 2 and 'provenance' in doc
        assert 'provenance' in json.load(open(tmp_path / 'export_manifest.json'))
        try:
            X.export(Project(), str(tmp_path), ['_nope'])
        except StudyError as e:
            assert 'no export format' in str(e)
        else:
            raise AssertionError
    finally:
        X.EXPORTERS.pop('_t_ok', None)
        X.EXPORTERS.pop('_t_no', None)
