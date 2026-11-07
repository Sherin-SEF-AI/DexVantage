"""Phase 1 product review: the flows a robotics researcher meets first, run headless, with the fixes the review made.

- Import errors say what is wrong in plain words: a xacro file (also one saved as .urdf), a URDF with no link, a folder,
  a USD file, a COLLADA mesh MuJoCo cannot decode; the wizard's mesh step does not call an unparsable file "missing".
- Trajectory CSVs with semicolons and decimal commas read as numbers; the degrees warning names its unit.
- --reproduce on a sample result does not report a variant difference that is only the robot's random uid.
- File ▸ Open sample; selecting a robot fills the Inspector; a robot and a trajectory can be deleted (one undo step);
  the Library no longer says robots arrive in Phase 1.

References: the expected values are written out here (the CSV's numbers, the counts of stored files), not taken from
the code under test."""
import json
import struct

import pytest
from PyQt6.QtWidgets import QLabel, QMessageBox

from conftest import pump
from dexvantage.core.project import ProjectFormatError
from dexvantage.core.robot import import_robot
from dexvantage.core.robot.filecheck import check_references
from dexvantage.core.robot.trajectory import read_csv

ARM = '''<?xml version="1.0"?>
<robot name="arm">
  <link name="base"><visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual></link>
  <link name="l1"><inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial><visual><geometry><mesh filename="{mesh}"/></geometry></visual></link>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" velocity="1" effort="1"/></joint>
</robot>
'''


def stl_box(h=0.05) -> bytes:
    c = [(x, y, z) for x in (-h, h) for y in (-h, h) for z in (-h, h)]
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = [t for a, b, cc, d in quads for t in ((a, b, cc), (a, cc, d))]
    out = b'\0' * 80 + struct.pack('<I', len(tris))
    for t in tris:
        out += struct.pack('<3f', 0, 0, 0) + b''.join(struct.pack('<3f', *c[i]) for i in t) + b'\0\0'
    return out


def write(tmp_path, name, text=None, data=None) -> str:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data) if data is not None else p.write_text(text)
    return str(p)


@pytest.fixture
def no_dialogs(monkeypatch):
    """Record QMessageBox calls instead of blocking on them."""
    seen = []
    for k in ('warning', 'information', 'critical'):
        monkeypatch.setattr(QMessageBox, k, lambda *a, _k=k, **kw: seen.append((_k, a[1], a[2])) or
                            QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, 'question', lambda *a, **k: QMessageBox.StandardButton.Discard)
    return seen


# ====================================================================== import errors
@pytest.mark.parametrize('name,text,words', [
    ('x.urdf.xacro', '<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="r"><link name="a"/></robot>', 'xacro'),
    ('macro.urdf', '<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="r"><xacro:include filename="a.xacro"/>'
                   '<link name="a"/></robot>', 'xacro file'),
    ('subst.urdf', '<robot name="r"><link name="a"><visual><geometry><mesh filename="$(find pkg)/m.stl"/></geometry>'
                   '</visual></link></robot>', 'xacro file'),
    ('nolink.urdf', '<robot name="r"></robot>', 'no <link>'),
    ('arm.usda', '#usda 1.0\n', 'USD file'),
])
def test_import_explains_files_it_cannot_read(tmp_path, name, text, words):
    path = write(tmp_path, name, text)
    with pytest.raises(ProjectFormatError, match=words):
        import_robot(path)
    with pytest.raises(ProjectFormatError, match=words):
        check_references(path)


def test_a_folder_is_called_a_folder(tmp_path):
    with pytest.raises(ProjectFormatError, match='is a folder: pick the robot'):
        import_robot(str(tmp_path))


def test_collada_mesh_says_convert_it(tmp_path):
    path = write(tmp_path, 'dae/arm.urdf', ARM.format(mesh='l1.dae'))
    write(tmp_path, 'dae/l1.dae', '<COLLADA/>')
    with pytest.raises(ProjectFormatError, match=r'l1\.dae, a COLLADA \(\.dae\) mesh, which MuJoCo cannot read\. '
                                                 'Convert the mesh to STL or OBJ'):
        import_robot(path)


def test_plain_urdf_still_imports(tmp_path):
    path = write(tmp_path, 'ok/arm.urdf', ARM.format(mesh='l1.stl'))
    write(tmp_path, 'ok/l1.stl', data=stl_box())
    robot, sources = import_robot(path)
    assert [ln['name'] for ln in robot.links] == ['base', 'l1'] and len(sources) == 3   # urdf, mesh, manifest


def test_wizard_does_not_call_an_unparsable_mjcf_missing(app, win, tmp_path):
    from dexvantage.ui.dialogs.import_wizard import ImportWizard
    path = write(tmp_path, 'bad.xml', '<mujoco><worldbody><body name="a"><geom size="0.1"/></body><body name="a"/>'
                                      '</worldbody></mujoco>')
    chk = check_references(path)
    assert [x['what'] for x in chk['items']] == ['model']
    w = ImportWizard(win.ctx, win, soft_3d=True)
    try:
        res = w.load(path, background=False)
        assert 'repeated name' in res['error']
        w.go(1)
        text = w.m_status.text.text()
        assert 'missing' not in text and 'cannot be listed' in text
        assert w.m_table.item(0, 2).text() == 'not read'
    finally:
        w.release()
        w.deleteLater()


def test_wizard_counts_in_words(app, win, tmp_path):
    from dexvantage.ui.dialogs.import_wizard import ImportWizard
    path = write(tmp_path, 'one/arm.urdf', ARM.format(mesh='l1.stl'))
    write(tmp_path, 'one/l1.stl', data=stl_box())
    w = ImportWizard(win.ctx, win, soft_3d=True)
    try:
        w.load(path, background=False)
        assert w.f_status.text.text().startswith('Read: 2 links, 1 moving joint')
        w.go(1)
        assert w.m_status.text.text() == 'The required file was found.'
        w.go(2)
        assert w.s_box.text().startswith('100 × 100 × 100 mm   (x −50 … +50, y −50 … +50, z −50 … +50)')
    finally:
        w.release()
        w.deleteLater()


# ====================================================================== trajectories
ROBOT = [('r1', ['shoulder', 'elbow'], [-2.0, -2.0], [2.0, 2.0], ['revolute', 'revolute'])]


def test_semicolon_csv_with_decimal_commas():
    rec, warn = read_csv('t;shoulder;elbow\n0;0,25;-0,5\n0,5;1,0;0\n', ROBOT)
    assert rec['t'] == [0.0, 0.5] and rec['q'] == [[0.25, -0.5], [1.0, 0.0]] and warn == []


def test_degrees_warning_names_its_unit():
    _rec, warn = read_csv('t,shoulder\n0,0\n1,90\n', [('r', ['shoulder'], [float('-inf')], [float('inf')],
                                                        ['continuous'])])
    assert warn and warn[0].startswith('shoulder reaches 90.0 rad: more than a full turn')


# ====================================================================== batch
def test_sample_result_reproduces_without_a_variant_difference(tmp_path, capsys):
    from dexvantage.app import main
    out = str(tmp_path / 'r.json')
    assert main(['--sample', 'ur5e', '--analyse', out]) == 0
    assert 'UR5e + Robotiq 2F-85 sample: 1 task volume, ' in capsys.readouterr().out
    assert json.load(open(out))['provenance']['variant']['sha256']
    assert main(['--sample', 'ur5e', '--reproduce', out]) == 0
    text = capsys.readouterr().out
    assert text.startswith('Reproduces exactly') and 'variant differs' not in text


# ====================================================================== the window
def test_open_sample_from_the_file_menu(app, win, no_dialogs):
    acts = {a.text(): a for a in win.m_samples.actions()}
    assert acts['UR5e and Robotiq 2F-85 with a wrist D405 and an overhead D435'].isEnabled()
    assert not acts['ALOHA 2 bimanual'].isEnabled() and 'Phase 11' in acts['ALOHA 2 bimanual'].toolTip()
    assert win.open_sample('so100') and win.store.project.name == 'SO-100 sample'
    assert not win.open_sample('aloha') and 'arrives in Phase 11' in no_dialogs[-1][2]


def test_robot_inspector_and_delete(app, win, no_dialogs, tmp_path):
    from dexvantage.ui.pages.robot import import_trajectory
    win.open_sample('ur5e')
    st = win.store
    r = st.project.primary_robot
    n_src = len(st.project.sources)
    assert n_src > 2
    st.select(('robot', r.uid))
    pump(app, 0.02)
    texts = [lab.text() for lab in win.inspector.findChildren(QLabel)]
    assert 'Moving joints' in texts and 'gripper_base' in texts and 'primary' in texts
    assert win.sel_label.text() == f'Selected: {r.name}'
    win.delete_selection()                                   # refused: a sensor rides on the wrist
    assert st.project.robots and no_dialogs[-1][1] == 'Robot not deleted' and 'Wrist D405' in no_dialogs[-1][2]
    csv = write(tmp_path, 'q.csv', 't,shoulder_pan_joint\n0,0\n1,0.5\n')
    assert import_trajectory(win.ctx, csv)
    for s in list(st.project.variant.sensors):
        if s.mount.robot_uid == r.uid:
            st.select(('sensor', s.uid))
            win.delete_selection()
    st.select(('robot', r.uid))
    n = st.undo.count()
    win.delete_selection()
    assert st.project.robots == [] and st.project.sources == {} and st.project.tasks.trajectories == []
    assert st.undo.count() == n + 1 and r.uid not in st.q
    st.undo.undo()
    assert [x.uid for x in st.project.robots] == [r.uid] and len(st.project.sources) == n_src
    assert len(st.project.tasks.trajectories) == 2 and len(st.q[r.uid]) == 14      # the sample's pick and q.csv


def test_trajectory_select_play_and_delete(app, win, no_dialogs, tmp_path):
    from dexvantage.ui.pages.robot import import_trajectory
    win.open_sample('so100')
    st = win.store
    for name, v in (('first', 0.2), ('second', -0.3)):
        assert import_trajectory(win.ctx, write(tmp_path, f'{name}.csv', f't,Rotation\n0,0\n1,{v}\n'))
    first, second = st.project.tasks.trajectories
    tree = win.outliner.tree
    tasks = next(tree.topLevelItem(i) for i in range(tree.topLevelItemCount()) if tree.topLevelItem(i).text(0) ==
                 'Tasks')
    assert [tasks.child(i).text(0) for i in range(tasks.childCount())] == ['first', 'second']
    assert tasks.child(0).text(1) == '2 · 1.00 s'
    st.select(('trajectory', first['uid']))
    assert win.scrubber.current()['uid'] == first['uid']
    win.scrubber.scrub_to(1.0)
    assert abs(st.q[st.project.primary_robot.uid][0] - 0.2) < 1e-12
    texts = [lab.text() for lab in win.inspector.findChildren(QLabel)]
    assert 'Samples' in texts and '1.000 s' in texts and '1: Rotation' in texts
    win.delete_selection()
    assert [t['name'] for t in st.project.tasks.trajectories] == ['second']
    assert win.scrubber.current()['uid'] == second['uid']
    st.undo.undo()
    assert len(st.project.tasks.trajectories) == 2


def test_library_and_robot_readouts(app, win):
    tree = win.library.tree
    groups = {tree.topLevelItem(i).text(0): tree.topLevelItem(i) for i in range(tree.topLevelItemCount())}
    for g in ('Robots', 'Grippers'):
        kids = [groups[g].child(k).text(0) for k in range(groups[g].childCount())]
        assert kids and not any('Arrives in Phase' in k for k in kids)
    assert groups['Cameras'].childCount() and 'Arrives in Phase' not in groups['Cameras'].child(0).text(0)   # Phase 2
    win.open_sample('ur5e')
    win.open_page('robot')
    pump(app, 0.05)
    page = win.pages['robot']
    page.refresh_readouts()
    panel = next(iter(page._panels.values()))
    assert '−0.0' not in panel.rot.findChildren(QLabel)[0].text() and '−0.0' not in panel.pos.findChildren(
        QLabel)[0].text()


def test_a_robot_file_on_the_command_line_is_explained(tmp_path, capsys):
    from dexvantage.app import main
    path = write(tmp_path, 'arm.urdf', ARM.format(mesh='l1.stl'))
    assert main([path, '--analyse', '-']) == 2
    assert 'arm.urdf is a robot model file, not a project (.dexv)' in capsys.readouterr().err
