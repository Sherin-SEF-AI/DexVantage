"""MuJoCo Menagerie listing, download, licence and import (spec §1.2, §9.4), with the network mocked: every request
goes to a fake opener that serves a made-up repository, and an unknown URL fails the test."""
import io
import json
import os
import socket
import urllib.error

import pytest

from dexvantage.core.robot import menagerie as MG

APACHE = ('                                 Apache License\n                           Version 2.0, January 2004\n'
          '                        http://www.apache.org/licenses/\n\n   TERMS AND CONDITIONS FOR USE ...\n')
BSD3 = ('Copyright 2018 Someone\n\nRedistribution and use in source and binary forms, with or without modification, '
        'are permitted provided that ...\n3. Neither the name of the copyright holder nor the names of its '
        'contributors may be used to endorse or promote products ...\n')
BSD2 = ('Copyright (c) 2013, Someone\nAll rights reserved.\n\nRedistribution and use in source and binary forms, with '
        'or without modification, are permitted provided that the following conditions are met: ...\n')
MIT = 'MIT License\n\nPermission is hereby granted, free of charge, to any person obtaining a copy ...\n'
CC_NC = ('Attribution-NonCommercial 4.0 International\n\nCreative Commons Corporation ("Creative Commons") is not a '
         'law firm ...\n')
# excerpts of the real texts, keeping the wording that used to mislabel them
GPL3 = ('GNU GENERAL PUBLIC LICENSE\nVersion 3, 29 June 2007\n... c) Convey individual copies of the object code with '
        'a copy of the written offer to provide the Corresponding Source. This alternative is allowed only '
        'occasionally and noncommercially ...\n13. Use with the GNU Affero General Public License. ...\n')
LGPL3 = ('GNU LESSER GENERAL PUBLIC LICENSE\nVersion 3, 29 June 2007\n... As used herein, "this License" refers to '
         'version 3 of the GNU Lesser General Public License, and the "GNU GPL" refers to version 3 of the GNU General '
         'Public License.\n')
CLEAR_BSD = ('The Clear BSD License\n\nCopyright (c) 2022 Someone\nAll rights reserved.\n\nRedistribution and use in '
             'source and binary forms ... Neither the name of the copyright holder nor the names of its contributors '
             'may be used to endorse or promote products ... NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY\'S PATENT '
             'RIGHTS ARE GRANTED BY THIS LICENSE.\n')
CC0 = ('Creative Commons Legal Code\n\nCC0 1.0 Universal\n\n    CREATIVE COMMONS CORPORATION IS NOT A LAW FIRM ...\n')
CC_BY = ('Attribution 4.0 International\n\nCreative Commons Corporation ("Creative Commons") is not a law firm ...\n'
         'Creative Commons may elect to apply one of its public licenses to material it publishes ... the CC0 Public '
         'Domain Dedication ...\n')
OFL = ('Copyright 2017 IBM Corp.\n\nThis Font Software is licensed under the SIL Open Font License, Version 1.1.\n...\n'
       'Permission is hereby granted, free of charge, to any person obtaining a copy of the Font Software ...\n')

ARM_XML = b"""<mujoco model="fake_arm">
  <compiler angle="radian" meshdir="assets"/>
  <asset><mesh name="link" file="link.stl"/></asset>
  <worldbody>
    <body name="base">
      <geom type="box" size="0.05 0.05 0.02"/>
      <body name="link1" pos="0 0 0.05">
        <joint name="j1" axis="0 0 1" range="-1.5 1.5"/>
        <geom type="mesh" mesh="link"/>
        <body name="tip" pos="0 0 0.2">
          <joint name="j2" axis="0 1 0" range="-2 2"/>
          <geom type="capsule" size="0.01 0.05"/>
        </body>
      </body>
    </body>
  </worldbody>
  <keyframe><key name="home" qpos="0.25 -0.5"/></keyframe>
</mujoco>
"""
SCENE_XML = b'<mujoco model="scene"><include file="fake_arm.xml"/><worldbody><light pos="0 0 2"/></worldbody></mujoco>'
COMMIT = 'abcdef0123456789abcdef0123456789abcdef01'


def _stl() -> bytes:
    """A small closed tetrahedron, binary STL (metres)."""
    import struct
    v = [(0, 0, 0), (0.04, 0, 0), (0, 0.04, 0), (0, 0, 0.04)]
    faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    out = io.BytesIO()
    out.write(b'test'.ljust(80, b' '))
    out.write(struct.pack('<I', len(faces)))
    for f in faces:
        out.write(struct.pack('<3f', 0, 0, 0))
        for i in f:
            out.write(struct.pack('<3f', *v[i]))
        out.write(b'\0\0')
    return out.getvalue()


class _Resp(io.BytesIO):
    def __init__(self, data, headers=None):
        super().__init__(data)
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class FakeGitHub:
    """Serves a made-up Menagerie: folders fake_arm (with LICENSE) and nolicence (without)."""

    def __init__(self, license_text=APACHE, files=None, fail=None):
        self.calls = []
        self.fail = fail or {}                         # url substring -> exception to raise
        self.files = files if files is not None else {
            'fake_arm/LICENSE': license_text.encode(),
            'fake_arm/fake_arm.xml': ARM_XML,
            'fake_arm/scene.xml': SCENE_XML,
            'fake_arm/assets/link.stl': _stl(),
            'fake_arm/fake_arm.png': b'\x89PNG gallery picture',
            'fake_arm/README.md': b'# readme',
            'fake_arm/../evil.txt': b'outside',
            'nolicence/thing.xml': ARM_XML,
        }

    def tree(self, folder):
        out = []
        for p, b in self.files.items():
            if p.startswith(folder + '/'):
                out.append({'path': p[len(folder) + 1:], 'type': 'blob', 'size': len(b), 'sha': 'x'})
        return out

    def __call__(self, url):
        self.calls.append(url)
        for k, e in self.fail.items():
            if k in url:
                raise e
        api = MG.API
        if url == f'{api}/contents/?ref={MG.BRANCH}':
            dirs = sorted({p.split('/')[0] for p in self.files})
            body = [{'name': d, 'type': 'dir'} for d in dirs] + [{'name': 'assets', 'type': 'dir'},
                                                                 {'name': 'README.md', 'type': 'file'}]
            return _Resp(json.dumps(body).encode())
        if url == f'{api}/commits/{MG.BRANCH}':
            return _Resp(json.dumps({'sha': COMMIT}).encode())
        if url == f'{api}/git/trees/{COMMIT}':
            dirs = sorted({p.split('/')[0] for p in self.files})
            return _Resp(json.dumps({'tree': [{'path': d, 'type': 'tree', 'sha': f'tree-{d}'} for d in dirs]}).encode())
        if url.startswith(f'{api}/git/trees/tree-') and url.endswith('?recursive=1'):
            folder = url[len(f'{api}/git/trees/tree-'):-len('?recursive=1')]
            return _Resp(json.dumps({'tree': self.tree(folder), 'truncated': False}).encode())
        pre = f'{MG.RAW}/{COMMIT}/'
        if url.startswith(pre) and url[len(pre):] in self.files:
            return _Resp(self.files[url[len(pre):]])
        raise AssertionError(f'the test asked for an unexpected URL: {url}')


# ====================================================================== licences
@pytest.mark.parametrize('text, name, nc', [(APACHE, 'Apache-2.0', False), (BSD3, 'BSD-3-Clause', False),
                                            (BSD2, 'BSD-2-Clause', False), (MIT, 'MIT', False),
                                            (CC_NC, 'CC-BY-NC-4.0', True),
                                            (GPL3, 'GPL-3.0', False), (LGPL3, 'LGPL-3.0', False),
                                            (CLEAR_BSD, 'BSD-3-Clause-Clear', False), (CC0, 'CC0-1.0', False),
                                            (CC_BY, 'CC-BY-4.0', False), (OFL, 'OFL-1.1', False),
                                            ('All rights reserved. Ask us.', 'see the LICENSE text', False)])
def test_license_name(text, name, nc):
    assert MG.license_name(text) == (name, nc)


# ====================================================================== the index
def test_index_from_github():
    gh = FakeGitHub()
    idx = MG.index(gh)
    assert idx['source'] == 'github'
    assert idx['models'] == ['fake_arm', 'nolicence']          # 'assets' and files are not models
    assert len(gh.calls) == 1


@pytest.mark.parametrize('err, words', [
    (urllib.error.URLError(socket.gaierror(-2, 'Name or service not known')), 'offline'),
    (urllib.error.URLError(TimeoutError('timed out')), 'did not answer'),
    (urllib.error.HTTPError(MG.API, 403, 'rate limited', {'X-RateLimit-Remaining': '0',
                                                          'X-RateLimit-Reset': '1900000000'}, None), '60 requests'),
    (urllib.error.HTTPError(MG.API, 500, 'Server Error', {}, None), 'answered 500'),
])
def test_index_falls_back_offline_and_explains(err, words):
    idx = MG.index(FakeGitHub(fail={'/contents/': err}))
    assert idx['source'] == 'offline'
    assert idx['models'] == sorted(MG.OFFLINE_MODELS)
    assert words in idx['note'] and 'Traceback' not in idx['note']
    assert 'trs_so_arm100' in idx['models'] and 'universal_robots_ur5e' in idx['models']


def test_offline_list_holds_names_only():
    assert all(isinstance(n, str) and n == n.strip() and '/' not in n for n in MG.OFFLINE_MODELS)
    assert len(set(MG.OFFLINE_MODELS)) == len(MG.OFFLINE_MODELS)


# ====================================================================== the download
def test_download_reads_the_licence_and_skips_gallery_files(tmp_path):
    gh = FakeGitHub()
    seen = []
    dl = MG.download('fake_arm', str(tmp_path), gh, progress=lambda f, t: seen.append(f))
    assert dl.commit == COMMIT and not dl.cached
    assert dl.license_text.startswith('                                 Apache License')
    assert (dl.license_name, dl.noncommercial) == ('Apache-2.0', False)
    assert dl.mjcf == ['fake_arm.xml', 'scene.xml']                # the robot file before the scene
    assert sorted(dl.files) == ['LICENSE', 'assets/link.stl', 'fake_arm.xml', 'scene.xml']
    assert dl.folder == os.path.join(str(tmp_path), 'fake_arm', COMMIT[:12])
    assert os.path.isfile(os.path.join(dl.folder, 'assets', 'link.stl'))
    assert not os.path.exists(os.path.join(str(tmp_path), 'fake_arm', 'evil.txt'))
    assert not os.path.exists(os.path.join(str(tmp_path), 'evil.txt'))
    assert seen and seen[-1] == 1.0
    assert COMMIT in dl.url
    # every file came from the pinned commit
    raw = [u for u in gh.calls if u.startswith(MG.RAW)]
    assert raw and all(f'/{COMMIT}/' in u for u in raw)
    # a second download of the same commit uses the files on disk
    gh2 = FakeGitHub()
    again = MG.download('fake_arm', str(tmp_path), gh2)
    assert again.cached and again.folder == dl.folder
    assert not [u for u in gh2.calls if u.startswith(MG.RAW)]


def test_download_offline_uses_an_earlier_copy(tmp_path):
    MG.download('fake_arm', str(tmp_path), FakeGitHub())
    off = FakeGitHub(fail={'api.github.com': urllib.error.URLError(socket.gaierror(-3, 'no network'))})
    dl = MG.download('fake_arm', str(tmp_path), off)
    assert dl.cached and dl.license_name == 'Apache-2.0'


def test_download_refuses_a_model_without_a_licence(tmp_path):
    with pytest.raises(MG.MenagerieError, match='no LICENSE file of its own'):
        MG.download('nolicence', str(tmp_path), FakeGitHub())
    assert not os.path.exists(os.path.join(str(tmp_path), 'nolicence'))


def test_download_network_failure_is_plain_and_leaves_nothing(tmp_path):
    gh = FakeGitHub(fail={'assets/link.stl': urllib.error.URLError(socket.gaierror(-2, 'Name or service not known'))})
    with pytest.raises(MG.MenagerieError) as ei:
        MG.download('fake_arm', str(tmp_path), gh)
    msg = str(ei.value)
    assert 'raw.githubusercontent.com' in msg and 'internet connection' in msg
    root = os.path.join(str(tmp_path), 'fake_arm')
    assert not os.listdir(root)                                # no half-written folder


def test_download_without_network_and_no_copy_is_plain(tmp_path):
    gh = FakeGitHub(fail={'api.github.com': urllib.error.URLError(ConnectionRefusedError(111, 'refused'))})
    with pytest.raises(MG.MenagerieError, match='could not reach api.github.com'):
        MG.download('fake_arm', str(tmp_path), gh)


def test_download_rejects_odd_names_and_missing_models(tmp_path):
    for bad in ('', '../x', 'a/b', 'assets'):
        with pytest.raises(MG.MenagerieError, match='not a MuJoCo Menagerie model'):
            MG.download(bad, str(tmp_path), FakeGitHub())
    with pytest.raises(MG.MenagerieError, match='no model named ghost'):
        MG.download('ghost', str(tmp_path), FakeGitHub())


def test_download_size_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(MG, 'MAX_DOWNLOAD_MB', 0.0001)
    with pytest.raises(MG.MenagerieError, match='more than DexVantage downloads'):
        MG.download('fake_arm', str(tmp_path), FakeGitHub())


def test_data_dir_follows_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path))
    assert MG.data_dir() == os.path.join(str(tmp_path), 'DexVantage', 'menagerie')


# ====================================================================== the import
def test_import_records_the_licence(tmp_path):
    dl = MG.download('fake_arm', str(tmp_path), FakeGitHub(license_text=BSD3))
    robot, sources = MG.import_download(dl, accepted_on='2026-11-06')
    assert robot.source.license == f'BSD-3-Clause (MuJoCo Menagerie fake_arm/LICENSE at {COMMIT[:12]}; accepted ' \
                                   '2026-11-06)'
    assert sources[f'{robot.uid}/LICENSE'].decode() == BSD3
    assert COMMIT in robot.source.note
    assert robot.home == {'j1': 0.25, 'j2': -0.5}
    assert any(k.endswith('assets/link.stl') for k in sources)


# ====================================================================== the dialog
def _dialog(win, tmp_path, gh):
    from dexvantage.ui.dialogs.menagerie import MenagerieDialog
    d = MenagerieDialog(win, opener=gh, dest=str(tmp_path), background=False)
    d.load_index()
    return d


def test_window_has_the_command(win):
    a = win.actions_by_id['import_menagerie']
    assert a.isEnabled() and 'Menagerie' in a.text()
    assert a in win.menus['Robot'].actions()


def test_dialog_needs_explicit_consent_then_imports_one_undo_step(app, win, tmp_path):
    gh = FakeGitHub(license_text=BSD2)
    d = _dialog(win, tmp_path, gh)
    assert [d.view.item(i).text() for i in range(d.view.count())] == ['fake_arm', 'nolicence']
    assert d.select_model('fake_arm')
    assert not d.import_btn.isEnabled() and not d.accept_box.isEnabled()
    assert not d.import_selected()                             # nothing before the licence is shown and accepted
    assert not win.store.project.robots
    d.fetch_selected()
    assert d.lic_text.toPlainText() == BSD2
    assert 'BSD-2-Clause' in d.lic_name.text()
    assert d.accept_box.isEnabled() and not d.accept_box.isChecked()
    assert not d.import_btn.isEnabled()
    assert not d.import_selected()
    d.accept_box.setChecked(True)
    assert d.import_btn.isEnabled()
    n_undo = win.store.undo.count()
    assert d.import_selected()
    p = win.store.project
    assert len(p.robots) == 1 and p.robots[0].primary
    assert p.robots[0].source.license.startswith('BSD-2-Clause (MuJoCo Menagerie fake_arm/LICENSE')
    assert f'{p.robots[0].uid}/LICENSE' in p.sources
    assert win.store.undo.count() == n_undo + 1
    win.store.undo.undo()
    assert not win.store.project.robots
    assert not any(k.endswith('/LICENSE') for k in win.store.project.sources)
    d.deleteLater()


def test_dialog_consent_resets_when_the_model_changes(win, tmp_path):
    d = _dialog(win, tmp_path, FakeGitHub())
    d.select_model('fake_arm')
    d.fetch_selected()
    d.accept_box.setChecked(True)
    assert d.import_btn.isEnabled()
    d.select_model('nolicence')
    assert not d.accept_box.isChecked() and not d.import_btn.isEnabled()
    d.fetch_selected()                                         # no LICENSE: refused, explained, nothing to accept
    assert 'no LICENSE file of its own' in d.status.text.text()
    assert not d.accept_box.isEnabled() and not d.import_btn.isEnabled()
    d.deleteLater()


def test_dialog_offline_and_noncommercial_warnings(win, tmp_path):
    off = FakeGitHub(fail={'/contents/': urllib.error.URLError(socket.gaierror(-2, 'no network'))})
    d = _dialog(win, tmp_path, off)
    assert 'built into DexVantage' in d.index_note.text()
    assert d.view.count() == len(MG.OFFLINE_MODELS)
    d.deleteLater()
    nc = FakeGitHub(license_text=CC_NC)
    nc.files = {k: v for k, v in nc.files.items()}
    d = _dialog(win, tmp_path, nc)
    d.select_model('fake_arm')
    d.fetch_selected()
    assert 'commercial' in d.status.text.text() and 'Restricted' in d.status.text.text()
    d.deleteLater()


def test_dialog_import_error_is_plain(win, tmp_path):
    gh = FakeGitHub()
    gh.files['fake_arm/fake_arm.xml'] = b'<mujoco><worldbody><body><geom type="mesh" mesh="nope"/></body>' \
                                        b'</worldbody></mujoco>'
    d = _dialog(win, tmp_path, gh)
    d.select_model('fake_arm')
    d.fetch_selected()
    d.accept_box.setChecked(True)
    assert not d.import_selected()
    assert d.status.text.text().startswith('Not imported:') and 'Traceback' not in d.status.text.text()
    assert not win.store.project.robots
    d.deleteLater()


def test_dialog_background_thread_delivers_on_the_gui_thread(app, win, tmp_path):
    import time
    from dexvantage.ui.dialogs.menagerie import MenagerieDialog
    d = MenagerieDialog(win, opener=FakeGitHub(), dest=str(tmp_path), background=True)
    d.load_index()
    t = time.time()
    while d.view.count() == 0 and time.time() - t < 5:
        app.processEvents()
        time.sleep(0.005)
    assert d.view.count() == 2
    d.select_model('fake_arm')
    d.fetch_selected()
    t = time.time()
    while d.dl is None and time.time() - t < 5:
        app.processEvents()
        time.sleep(0.005)
    assert d.dl is not None and d.accept_box.isEnabled() and not d.import_btn.isEnabled()
    d.deleteLater()
