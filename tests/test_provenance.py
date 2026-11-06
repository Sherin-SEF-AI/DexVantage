import json
import os

import pytest

from dexvantage.core import provenance as PV
from dexvantage.core.project import blank_project
from dexvantage.validation.checks_formats import reference_project


def test_canonical_treats_integral_floats_as_integers():
    assert PV.canonical({'a': 2.0, 'b': [1, 2.5]}) == PV.canonical({'b': [1.0, 2.5], 'a': 2})


def test_result_hash_ignores_times_and_provenance():
    a = {'x': 1, 'seconds': 0.5, 'when': '2026-11-06T10:00:00Z'}
    b = {'x': 1, 'seconds': 9.0, 'when': '2026-01-01T00:00:00Z', 'provenance': {'any': 1}}
    assert PV.result_digest(a) == PV.result_digest(b)
    assert PV.result_digest(a) != PV.result_digest({**a, 'x': 2})


def test_content_hash_ignores_uids_and_times():
    a, b = reference_project(), reference_project()
    assert a.robots[0].uid != b.robots[0].uid
    assert PV.project_digest(a)['sha256'] == PV.project_digest(b)['sha256']
    b.workspace.volumes[0].size = (0.4, 0.6, 0.3)
    da, db = PV.project_digest(a), PV.project_digest(b)
    assert da['sha256'] != db['sha256']
    assert [k for k in da['sections'] if da['sections'][k] != db['sections'][k]] == ['workspace']


def test_block_has_no_paths_of_this_machine(tmp_path):
    p = blank_project('Cell')
    p.path = str(tmp_path / 'deep' / 'cell.dexv')
    blk = PV.block(p, {'schema': 'dexvantage.summary/1', 'x': 1})
    text = json.dumps(blk)
    assert str(tmp_path) not in text and os.path.expanduser('~') not in text
    assert blk['project']['file'] == 'cell.dexv'
    assert blk['schema'] == 'dexvantage.provenance/1' and blk['dexvantage']['version']
    assert blk['runtime']['mujoco']


def test_stamp_and_verify_detect_tampering():
    doc = PV.stamp({'schema': 'dexvantage.summary/1', 'n': 3}, blank_project())
    assert list(doc)[-1] == 'provenance' and PV.verify(doc)['ok'] is True
    doc['n'] = 4
    assert PV.verify(doc)['ok'] is False


def test_stamp_bytes_keeps_the_file_formatting():
    text = '{\n  "schema": "dexvantage.summary/1",\n  "a": 1\n}\n'
    out = PV.stamp_bytes(text.encode(), blank_project()).decode()
    assert out.startswith('{\n  "schema": "dexvantage.summary/1",\n  "a": 1,\n  "provenance": {')
    assert PV.verify(json.loads(out))['ok'] is True


def test_scrub_reduces_paths_to_names(tmp_path):
    out = PV.scrub({'path': str(tmp_path / 'x' / 'file.json'), 'note': f'read {tmp_path}/y/z.csv today'})
    assert out['path'] == 'file.json' and str(tmp_path) not in out['note'] and 'z.csv' in out['note']


def test_compare_and_tolerances():
    a = {'v': 1.0, 'w': [1, 2], 't': 'x'}
    assert PV.compare(a, dict(a))['verdict'] == 'reproduces'
    b = {'v': 1.0 + 1e-10, 'w': [1, 2], 't': 'x'}
    assert PV.compare(a, b)['verdict'] == 'does_not_reproduce'
    assert PV.compare(a, b, PV.parse_tolerances(['1e-9']))['verdict'] == 'reproduces_within_tolerance'
    assert PV.compare(a, {**b, 't': 'y'}, PV.parse_tolerances(['1e-9']))['verdict'] == 'does_not_reproduce'
    with pytest.raises(ValueError):
        PV.parse_tolerances(['v=-1'])


def test_code_digest_covers_the_package():
    assert PV.code_digest() and len(PV.code_digest()) == 64
