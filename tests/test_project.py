import json
import os
import zipfile

import numpy as np
import pytest

from dexvantage.core import project as P
from dexvantage.core import provenance as PV
from dexvantage.validation.checks_formats import reference_project

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def test_round_trip_keeps_everything(tmp_path):
    p = reference_project()
    path = str(tmp_path / 'a.dexv')
    p.save(path, thumb=b'\x89PNG')
    q = P.Project.load(path)
    assert PV.project_digest(q) == PV.project_digest(p)
    assert q.thumb == b'\x89PNG' and q.path == path
    assert np.array_equal(q.meshes['l1/visual'], p.meshes['l1/visual'])
    with zipfile.ZipFile(path) as z:
        assert {'project.json', 'meshes.npz', 'sources/left.urdf', 'thumb.png'} <= set(z.namelist())


def test_newer_version_is_refused_with_both_versions_named(tmp_path):
    path = tmp_path / 'new.dexv'
    doc = P.blank_project().to_dict()
    doc['version'] = P.VERSION + 1
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('project.json', json.dumps(doc))
    with pytest.raises(P.ProjectFormatError) as e:
        P.Project.load(str(path))
    assert f'version {P.VERSION + 1}' in str(e.value) and f'version {P.VERSION}' in str(e.value)


@pytest.mark.parametrize('content, words', [(b'not a zip', 'not a DexVantage project'),
                                            (None, 'no project.json')])
def test_bad_files_are_explained(tmp_path, content, words):
    path = tmp_path / 'bad.dexv'
    if content is None:
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr('other.txt', 'x')
    else:
        path.write_bytes(content)
    with pytest.raises(P.ProjectFormatError) as e:
        P.Project.load(str(path))
    assert words in str(e.value)


def test_missing_file_is_explained(tmp_path):
    with pytest.raises(P.ProjectFormatError):
        P.Project.load(str(tmp_path / 'nope.dexv'))


def test_sections_restore():
    p = reference_project()
    for key in ('meta', 'robots', 'workspace', 'tasks', 'variants', 'requirements', 'analysis',
                f'variant:{p.active_uid}'):
        before = PV.project_digest(p)['sha256']
        p.restore(key, p.section(key))
        assert PV.project_digest(p)['sha256'] == before, key


def test_published_schema_is_the_packaged_one():
    with open(os.path.join(ROOT, 'docs', 'project.schema.json')) as a, \
            open(os.path.join(ROOT, 'dexvantage', 'resources', 'project.schema.json')) as b:
        assert json.load(a) == json.load(b)


def test_blank_project_validates_against_schema():
    jsonschema = pytest.importorskip('jsonschema', reason='jsonschema (the test extra) is not installed')
    with open(os.path.join(ROOT, 'docs', 'project.schema.json')) as fh:
        schema = json.load(fh)
    jsonschema.validate(json.loads(json.dumps(P.blank_project().to_dict())), schema)


def test_mount_on_link_needs_a_robot():
    assert not P.Mount('tripod').on_link
    assert not P.Mount('wrist_link').on_link             # no robot named
    assert P.Mount('wrist_link', 'abcdef0123').on_link


def test_unknown_values_fall_back_safely():
    s = P.Sensor.from_dict({'kind': 'radar', 'mount': {'parent_frame': ''}})
    assert s.kind == 'camera' and s.mount.parent_frame == 'world'
    o = P.TaskObject.from_dict({'materials': ['glossy', 'chrome']})
    assert o.materials == ['glossy']
