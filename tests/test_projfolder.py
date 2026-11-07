"""The unpacked, Git-friendly project folder (core/projfolder.py), Project.load/save on folders, --unpack, --pack and
--textconv."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from dexvantage.core import projdiff as D, projfolder as PF, provenance as PV
from dexvantage.core.project import Project, ProjectFormatError
from dexvantage.core.robot.samples import sample_project

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def _run(*args, cwd=None):
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    return subprocess.run([sys.executable, '-m', 'dexvantage', *args], capture_output=True, text=True, env=env,
                          cwd=cwd or ROOT, timeout=120)


def _files(root) -> dict:
    out = {}
    for dp, _dn, fn in os.walk(root):
        for f in fn:
            p = os.path.join(dp, f)
            with open(p, 'rb') as fh:
                out[os.path.relpath(p, root).replace(os.sep, '/')] = fh.read()
    return out


@pytest.fixture(scope='module')
def proj():
    p = sample_project('so100')
    p.meshes = {'object/mug verts': np.arange(36, dtype=np.float32).reshape(12, 3), 'faces': np.eye(3, dtype=np.int32)}
    p.thumb = b'\x89PNG fake'
    return p


def test_layout_and_pretty_sorted_json(proj, tmp_path):
    f = str(tmp_path / 'cell.dexv.d')
    PF.save_unpacked(proj, f)
    names = set(os.listdir(f))
    assert {'project.json', 'robots.json', 'variants.json', 'workspace.json', 'tasks.json', 'analysis.json',
            'policies.json', 'log.json', 'meshes', 'sources', 'thumb.png'} <= names
    head = json.loads(open(os.path.join(f, 'project.json'), encoding='utf-8').read())
    assert head['layout'] == 'dexvantage.unpacked/1' and 'saved_at' not in head
    for name in names:
        if name.endswith('.json'):
            text = open(os.path.join(f, name), encoding='utf-8').read()
            obj = json.loads(text)
            assert text == json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + '\n', name
    assert set(head['meshes'].values()) == {'meshes/faces.npy', next(v for v in head['meshes'].values()
                                                                     if 'mug' in v)}
    for name in proj.sources:
        assert os.path.isfile(os.path.join(f, 'sources', *name.split('/')))


def test_unpack_pack_round_trip_is_byte_stable(proj, tmp_path):
    a, b, c = (str(tmp_path / n) for n in ('a.dexv.d', 'b.dexv.d', 'c.dexv.d'))
    z = str(tmp_path / 'p.dexv')
    PF.save_unpacked(proj, a)
    q = Project.load(a)                                     # Project.load reads a folder
    PF.save_unpacked(q, b)
    assert _files(a) == _files(b)
    PF.pack(a, z)                                           # folder -> .dexv -> folder
    PF.unpack(z, c)
    assert _files(a) == _files(c)
    h = PV.project_digest(proj)['sha256']
    assert PV.project_digest(q)['sha256'] == h == PV.project_digest(Project.load(z))['sha256']
    for k, arr in proj.meshes.items():
        assert q.meshes[k].dtype == arr.dtype and np.array_equal(q.meshes[k], arr)
    assert q.sources == proj.sources and q.thumb == proj.thumb
    assert D.diff(proj, q)['identical']


def test_saving_again_writes_only_what_changed_and_keeps_other_files(proj, tmp_path):
    f = str(tmp_path / 'cell.dexv.d')
    PF.save_unpacked(proj, f)
    with open(os.path.join(f, '.gitattributes'), 'w') as fh:
        fh.write('*.npy binary\n')
    p = proj.snapshot()
    p.variant.sensors[0].name = 'Renamed'
    p.meshes = {'faces': proj.meshes['faces']}              # one mesh dropped
    r = PF.save_unpacked(p, f)
    assert sorted(r['written']) == ['project.json', 'variants.json']     # the mesh list is in project.json
    assert len(r['removed']) == 1 and 'mug' in r['removed'][0]
    assert os.path.isfile(os.path.join(f, '.gitattributes'))


def test_project_save_and_load_dispatch_on_folders(proj, tmp_path):
    f = str(tmp_path / 'x.dexv.d')
    p = proj.snapshot()
    p.save(f)                                               # the window's Save on a .dexv.d path
    assert os.path.isdir(f) and p.path == f
    q = Project.load(f)
    assert q.path == f and D.diff(p, q)['identical']


def test_unreadable_folders_are_explained(tmp_path, proj):
    with pytest.raises(ProjectFormatError, match='no project.json'):
        Project.load(str(tmp_path))
    f = str(tmp_path / 'y.dexv.d')
    PF.save_unpacked(proj, f)
    os.remove(os.path.join(f, 'robots.json'))
    with pytest.raises(ProjectFormatError, match='robots.json'):
        Project.load(f)
    bad = proj.snapshot()
    bad.sources = {'../escape.stl': b'x'}
    with pytest.raises(ProjectFormatError, match='leave the project folder'):
        PF.save_unpacked(bad, str(tmp_path / 'z.dexv.d'))
    PF.save_unpacked(proj, f)                               # restores robots.json
    head_path = os.path.join(f, 'project.json')
    head = json.load(open(head_path))
    head['sources'] = ['../../etc/passwd']
    PF._write_if_changed(head_path, PF.dumps(head).encode())
    with pytest.raises(ProjectFormatError, match='outside the folder'):
        PF.load_unpacked(f)


def test_textconv_is_stable_and_names_binaries(proj, tmp_path):
    z = str(tmp_path / 'p.dexv')
    proj.snapshot().save(z)
    t1 = PF.textconv(z)
    proj.snapshot().save(z)                                 # saved again: a new saved_at, the same text
    assert PF.textconv(z) == t1
    assert t1.startswith('### project.json\n') and '### robots.json\n' in t1
    assert '### meshes/faces.npy  <i4 [3, 3]  sha256 ' in t1
    assert '\x00' not in t1


def test_batch_unpack_pack_textconv(tmp_path):
    f = str(tmp_path / 's.dexv.d')
    z = str(tmp_path / 's.dexv')
    r = _run('--sample', 'so100', '--unpack', f)
    assert r.returncode == 0 and os.path.isfile(os.path.join(f, 'robots.json')), r.stderr
    r = _run(f, '--pack', z)
    assert r.returncode == 0 and os.path.isfile(z), r.stderr
    r = _run('--textconv', z)
    assert r.returncode == 0 and r.stdout == PF.textconv(z) and not r.stderr
    r = _run('--textconv', str(tmp_path / 'nothing.dexv'))
    assert r.returncode == 2 and 'Traceback' not in r.stderr
    r = _run(z, '--pack', str(tmp_path / 'out.dexv.d'))
    assert r.returncode == 2 and '--unpack' in r.stderr
