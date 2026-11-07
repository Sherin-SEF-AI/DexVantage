"""The robot catalogue (core/robot/catalogue.py, library kind 'robot'): every entry valid and sourced, its Menagerie
folder known, its DoF consistent, and the Library's Import flow with the network mocked (a fake opener serving the
bundled SO-ARM100 folder as if it were GitHub; an unknown URL fails the test)."""
import glob
import io
import json
import os
import re

import pytest

from dexvantage.core import library as L
from dexvantage.core.robot import catalogue as CAT
from dexvantage.core.robot import menagerie as MG

FOLDER = os.path.join(L.BUILTIN_DIR, 'robot')
FILES = sorted(glob.glob(os.path.join(FOLDER, '*.json')))
DATE = re.compile(r'^20[0-9]{2}-[01][0-9]-[0-3][0-9]$')
COMMIT = 'abcdef0123456789abcdef0123456789abcdef01'

# spec: the robots the catalogue covers at least (their Menagerie folders)
REQUIRED = {'franka_fr3', 'franka_emika_panda', 'kinova_gen3', 'kuka_iiwa_14', 'ufactory_xarm7', 'ufactory_lite6',
            'universal_robots_ur5e', 'universal_robots_ur10e', 'unitree_z1', 'trossen_vx300s', 'trossen_wx250s',
            'trs_so_arm100', 'low_cost_robot_arm', 'rethink_robotics_sawyer', 'leap_hand', 'wonik_allegro',
            'shadow_hand', 'robotiq_2f85', 'umi_gripper', 'unitree_g1', 'unitree_h1'}


def _load(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


@pytest.fixture(autouse=True)
def _fresh_library():
    L.reload()
    yield


# ------------------------------------------------------------------------------------------------ the entries
def test_the_catalogue_covers_the_robots_the_spec_names():
    assert len(FILES) >= len(REQUIRED)
    assert REQUIRED <= set(CAT.folders())
    types = {CAT.block(e)['type'] for e in CAT.entries()}
    assert {'arm', 'bimanual', 'hand', 'gripper', 'humanoid', 'mobile_manipulator'} <= types


@pytest.mark.parametrize('path', FILES, ids=os.path.basename)
def test_entry_matches_the_schema_and_its_file_name(path):
    d = _load(path)
    assert L.validate(d) == [], path
    assert L.validate_minimal(d) == [], path
    assert d['kind'] == 'robot' and os.path.basename(path) == d['id'] + '.json'
    assert d['robot']['type'] in CAT.ROBOT_TYPES


@pytest.mark.parametrize('path', FILES, ids=os.path.basename)
def test_every_figure_has_a_source_and_a_quote(path):
    """A figure is null (not stated) or names one of the entry's sources, dated, with the source's words."""
    d = _load(path)
    srcs = d['sources']
    for s in srcs:
        assert s['url'].startswith('https://') and DATE.match(s['read']) and s.get('title') and s.get('covers'), s
    for k in CAT.FIGURES:
        f = d['robot'].get(k)
        assert isinstance(f, dict), f'{k} missing (null figures are {{"value": null, "source": null}})'
        if f['value'] is None:
            assert f['source'] is None, k
            continue
        assert isinstance(f['source'], int) and 0 <= f['source'] < len(srcs), k
        assert f.get('quote', '').strip(), f'{k} has no quote from its source'
        if k == 'joint_ranges':
            assert f.get('unit') in ('deg', 'rad') and all(j['min'] <= j['max'] for j in f['value']), k
        elif k == 'interfaces':
            assert isinstance(f['value'], list) and all(isinstance(x, str) for x in f['value'])
        elif k == 'dof':
            v = f['value']
            assert (isinstance(v, int) and v > 0) or (isinstance(v, list) and len(v) == 2 and v[0] <= v[1])
        else:
            assert isinstance(f['value'], (int, float)) and f['value'] > 0, k


@pytest.mark.parametrize('path', FILES, ids=os.path.basename)
def test_model_folder_licence_and_verified_record(path):
    d = _load(path)
    m, v = d['robot']['model'], d['robot']['verified']
    assert m['source'] == 'menagerie'
    assert m['folder'] in MG.OFFLINE_MODELS, m['folder']
    assert m['commit'] == v['commit'] and len(m['commit']) == 40
    assert m['file'] == v['file'] and m['file'].endswith('.xml')
    assert m['url'].endswith(f'/{m["commit"]}/{m["folder"]}')
    # every model in the catalogue carries a licence DexVantage recognises, and none restricts commercial use
    assert m['license'] in ('Apache-2.0', 'BSD-3-Clause', 'BSD-2-Clause', 'BSD-3-Clause-Clear', 'MIT',
                            'Apache-2.0 AND BSD-3-Clause'), m['license']     # franka_fr3, fr3_v2, flexiv_rizon4
    assert DATE.match(v['date']) and DATE.match(m['license_read'])
    if v['imported']:
        assert len(v['joint_names']) == v['joints_in_q'] and v['links'] >= 1
    else:
        assert v['note'].strip()


@pytest.mark.parametrize('path', FILES, ids=os.path.basename)
def test_dof_is_consistent(path):
    e = CAT.get(_load(path)['id'])
    assert CAT.dof_problems(e) == []


def test_no_personal_paths_or_contact_details():
    for path in FILES:
        text = open(path, encoding='utf-8').read()
        for bad in ('/' + 'home/', '/' + 'Users/', '/' + 'tmp/', 'C:' + '\\' + 'Users'):
            assert bad not in text, path
        assert not re.search(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}', text), path


def test_the_loader_kept_every_entry():
    assert [p for p in L.problems() if 'robot' in p] == []
    assert {e.id for e in CAT.entries()} == {os.path.basename(p)[:-5] for p in FILES}
    assert all(CAT.block(e)['type'] == 'hand' for e in CAT.entries('hand'))
    with pytest.raises(CAT.CatalogueError, match='no robot type'):
        CAT.entries('quadruped')


def test_dof_problems_catch_inconsistent_records():
    e = CAT.get('franka-emika-panda')
    d = json.loads(json.dumps(e.data))
    bad = L.Entry(d)
    d['robot']['verified']['dof'] = 9                       # does not follow from 9 joints minus 2 fingers
    assert any('does not follow' in p for p in CAT.dof_problems(bad))
    d['robot']['verified']['dof'] = 9
    d['robot']['verified']['excluded_joints'] = []
    assert any('maker states 7' in p for p in CAT.dof_problems(bad))
    d['robot']['verified']['dof_differs'] = 'a test explanation'
    assert CAT.dof_problems(bad) == []
    d['robot']['verified']['excluded_joints'] = ['ghost']
    assert any('not joints of the model' in p for p in CAT.dof_problems(bad))
    assert CAT.dof_matches([23, 43], 29) and CAT.dof_matches([23, 43], 50) is False and CAT.dof_matches(None, 3) is None


def test_summary_and_description_show_not_stated_figures_as_a_dash():
    e = CAT.get('kuka-lbr-iiwa-14-r820')
    assert CAT.summary(e) == 'Arm · 7 DoF · 820 mm · 14 kg'
    rows = dict((k, t) for k, t, _s in CAT.describe(e))
    assert rows['Mass'] == '—' and rows['Repeatability'] == '—'
    assert rows['Model'].startswith('MuJoCo Menagerie kuka_iiwa_14/iiwa14.xml at 98a70c569c4c')


# ------------------------------------------------------------------------------------------------ the import flow
class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class FakeGitHub:
    """Serves the bundled trs_so_arm100 folder as Menagerie's trs_so_arm100 at COMMIT."""

    def __init__(self):
        from dexvantage.core.robot.samples import robot_file
        root = robot_file('trs_so_arm100')
        self.files = {}
        for dp, _d, fs in os.walk(root):
            for f in fs:
                p = os.path.join(dp, f)
                self.files['trs_so_arm100/' + os.path.relpath(p, root).replace(os.sep, '/')] = open(p, 'rb').read()
        self.calls = []

    def __call__(self, url):
        import json as _j
        self.calls.append(url)
        api = MG.API
        if url == f'{api}/commits/{MG.BRANCH}':
            return _Resp(_j.dumps({'sha': COMMIT}).encode())
        if url == f'{api}/git/trees/{COMMIT}':
            return _Resp(_j.dumps({'tree': [{'path': 'trs_so_arm100', 'type': 'tree', 'sha': 'tree-so'}]}).encode())
        if url == f'{api}/git/trees/tree-so?recursive=1':
            tree = [{'path': p.split('/', 1)[1], 'type': 'blob', 'size': len(b)} for p, b in self.files.items()]
            return _Resp(_j.dumps({'tree': tree}).encode())
        pre = f'{MG.RAW}/{COMMIT}/'
        if url.startswith(pre) and url[len(pre):] in self.files:
            return _Resp(self.files[url[len(pre):]])
        raise AssertionError(f'the test asked for an unexpected URL: {url}')


def test_import_entry_downloads_and_imports_the_catalogue_file(tmp_path):
    e = CAT.get('the-robot-studio-so-arm100')
    robot, sources, dl = CAT.import_entry(e, opener=FakeGitHub(), dest=str(tmp_path), accepted_on='2026-11-07')
    assert dl.license_name == CAT.model(e)['license'] == 'Apache-2.0'
    assert robot.source.license.startswith('Apache-2.0 (MuJoCo Menagerie trs_so_arm100/LICENSE')
    assert 'Robot catalogue entry the-robot-studio-so-arm100' in robot.source.note
    moving = [j for j in robot.joints if j['type'] in ('revolute', 'prismatic', 'continuous')]
    assert len(moving) == CAT.verified(e)['joints_in_q'] == CAT.verified(e)['dof']
    with pytest.raises(CAT.CatalogueError, match='no entry'):
        CAT.import_entry('no-such-robot')


def test_library_dock_lists_the_catalogue_and_imports_after_consent(app, win, tmp_path):
    from dexvantage.ui.dialogs import robot_catalogue as RC
    lib = win.library
    lib.refill()
    it = lib.item_for('the-robot-studio-so-arm100')
    assert it is not None and it.parent().text(0) == 'Robots'
    assert 'Degrees of freedom: —' in it.toolTip(0) and 'Apache-2.0' in it.toolTip(0)
    # the two import commands are still there
    kids = [it.parent().child(i).text(0) for i in range(it.parent().childCount())]
    assert 'Import from MuJoCo Menagerie…' in kids
    gh = FakeGitHub()
    d = RC.import_entry(win, 'the-robot-studio-so-arm100', opener=gh, dest=str(tmp_path), background=False,
                        run=False)
    assert [d.view.item(i).text() for i in range(d.view.count())] == ['trs_so_arm100']
    assert d.dl is not None and d.file.currentText() == 'so_arm100.xml'
    assert 'Apache-2.0' in d.lic_name.text() and d.lic_text.toPlainText().strip()
    assert not d.import_btn.isEnabled() and not d.import_selected()      # nothing without consent
    assert not win.store.project.robots
    d.accept_box.setChecked(True)
    n_undo = win.store.undo.count()
    assert d.import_selected()
    p = win.store.project
    assert len(p.robots) == 1 and p.robots[0].source.license.startswith('Apache-2.0 (MuJoCo Menagerie trs_so_arm100')
    assert win.store.undo.count() == n_undo + 1
    win.store.undo.undo()
    assert not win.store.project.robots
    d.deleteLater()


def test_validation_check_passes_on_the_catalogue():
    from dexvantage import validation as VA
    from dexvantage.validation import checks_robots, suite
    spec = next(s for s in suite.CHECKS if s.id == 'robots.catalogue_dof')
    out = checks_robots._catalogue_dof(None)
    assert out.error == 0, out.detail['problems']
    assert out.ok and len(out.detail['bundled_recounted']) >= 4
    assert {'trs_so_arm100/so_arm100.xml', 'universal_robots_ur5e/ur5e.xml', 'robotiq_2f85/2f85.xml',
            'shadow_hand/right_hand.xml'} <= {r['file'] for r in out.detail['bundled_recounted']}
    assert spec.tolerance == 0 and all(r in VA.REFERENCES for r in spec.refs)


def test_combined_apache_and_bsd_licence_is_named_in_full():
    """franka_fr3/LICENSE at 98a70c5 is the Apache-2.0 text followed by a BSD text (truthfulness review)."""
    apache = 'Apache License Version 2.0, January 2004 TERMS AND CONDITIONS'
    bsd = ('Software License Agreement (BSD License) Redistribution and use in source and binary forms, with or '
           'without modification, are permitted ... Neither the name of Willow Garage, Inc. nor the names of its '
           'contributors may be used to endorse or promote products')
    assert MG.license_name(apache)[0] == 'Apache-2.0'
    assert MG.license_name(apache + ' ---- ' + bsd)[0] == 'Apache-2.0 AND BSD-3-Clause'
    for eid in ('franka-research-3', 'franka-research-3-v2', 'flexiv-rizon-4'):
        assert CAT.model(CAT.get(eid))['license'] == 'Apache-2.0 AND BSD-3-Clause'
