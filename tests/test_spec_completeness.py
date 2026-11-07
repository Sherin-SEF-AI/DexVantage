"""The spec-completeness review of Phases 9 and 10 (spec §7.16, §11, §13): the batch names of §11 (--grasp-study,
--cable-flex, --mujoco-validate out/, --handeye-solve with a captures folder, --formats usd, fiftyone with --dataset),
the Canonical view page's depth range field, the Workspace page and the design log's notes."""
import json
import os

import numpy as np
import pytest

from dexvantage.app import main
from dexvantage.core import exporters as X
from dexvantage.core import jobs
from dexvantage.core import workspace_check as WC
from dexvantage.core.frames import Pose
from dexvantage.core.project import TaskObject, Volume, blank_project
from dexvantage.core.robot.samples import sample_project

from conftest import pump


# ------------------------------------------------------------------ §11 batch names
def test_spec_flag_names_are_aliases(capsys):
    with pytest.raises(SystemExit):
        main(['--help'])
    out = capsys.readouterr().out
    for flag in ('--grasp-study', '--cable-flex', '--mujoco-validate', '--teleop-overlap', '--sync', '--captures',
                 '--joints'):
        assert flag in out, flag


def test_a_study_given_a_folder_writes_flag_json_inside(tmp_path, capsys):
    out = tmp_path / 'out'
    assert main(['--sample', 'so100', '--vla-tokens', str(out) + os.sep]) == 0
    doc = json.loads((out / 'vla-tokens.json').read_text(encoding='utf-8'))
    assert doc['provenance']['output']['name'] == 'vla-tokens.json'
    # an existing folder without the separator too
    assert main(['--sample', 'so100', '--vla-tokens', str(out)]) == 0
    assert 'VLA token resolution →' in capsys.readouterr().out


def test_usd_is_the_isaac_export(tmp_path):
    assert X.resolve(['mjcf', 'usd', 'isaac']) == ['mjcf', 'isaac']
    assert main(['--sample', 'so100', '--export', str(tmp_path / 'x'), '--formats', 'usd']) == 0
    man = json.loads((tmp_path / 'x' / 'export_manifest.json').read_text(encoding='utf-8'))
    assert list(man['formats']) == ['isaac'] and man['formats']['isaac']['status'] == 'ok'
    assert any(f.endswith('.usda') for f in man['files'])
    man = X.export(sample_project('so100'), str(tmp_path / 'y'), ['usd'], {'usd': {}})
    assert man['formats']['isaac']['status'] == 'ok'


def test_fiftyone_export_takes_the_dataset_from_batch_or_the_project(tmp_path, capsys):
    not_a_dataset = tmp_path / 'empty'
    not_a_dataset.mkdir()
    # --dataset with --export fiftyone is accepted and reaches the exporter (which refuses this folder in words)
    code = main(['--sample', 'so100', '--dataset', str(not_a_dataset), '--export', str(tmp_path / 'x'),
                 '--formats', 'fiftyone'])
    err = capsys.readouterr().err
    assert code == 2 and 'not a dataset format' in err and 'Traceback' not in err
    # no dataset anywhere: the sentence says where to give one
    man = X.export(sample_project('so100'), str(tmp_path / 'y'), ['fiftyone'])
    assert man['formats']['fiftyone']['status'] == 'refused'
    assert '--dataset PATH' in man['formats']['fiftyone']['reason']
    # the project's audit dataset is the default
    from dexvantage.core.studies.audit import use_dataset
    p = sample_project('so100')
    use_dataset(p, str(not_a_dataset))
    man = X.export(p, str(tmp_path / 'z'), ['fiftyone'])
    assert 'not a dataset format' in man['formats']['fiftyone']['reason']


def test_captures_needs_handeye_solve_and_a_joints_file(tmp_path, capsys):
    assert main(['--sample', 'ur5e', '--captures', str(tmp_path)]) == 2
    assert 'go with --handeye-solve' in capsys.readouterr().err
    assert main(['--sample', 'ur5e', '--handeye-solve', '-', '--captures', str(tmp_path)]) == 2
    assert 'has no joint states file' in capsys.readouterr().err
    assert main(['--sample', 'ur5e', '--handeye-solve', '-', '--captures', str(tmp_path / 'nope')]) == 2
    assert 'is not a folder' in capsys.readouterr().err


def test_handeye_solve_of_a_captures_folder(tmp_path, capsys):
    """The specification's `--handeye-solve captures/`: a simulated capture (MuJoCo renders of the planned poses with
    a known mount offset) solved in batch from its folder."""
    from dexvantage.core.calib import simcapture as SIM
    p = sample_project('ur5e')
    p.analysis.studies['handeye_plan'] = {'n_poses': 10}
    plan = jobs.run_study('handeye_plan', p)
    sim = SIM.simulate(p, plan['poses'], str(tmp_path / 'cap'), plan['camera']['sensor_uid'])
    out = tmp_path / 'solve.json'
    code = main(['--sample', 'ur5e', '--handeye-solve', str(out), '--captures', sim['images'], '--joints',
                 sim['joints']])
    assert code == 0, capsys.readouterr().err
    doc = json.loads(out.read_text(encoding='utf-8'))
    assert doc['ok'] and any('Captured at run time' in n for n in doc['notes'])
    # the solve finds the simulated offset (its 1σ is about 0.01 mm here; 0.3 mm leaves room for the image noise)
    assert abs(doc['apply']['change_mm'] - sim['truth']['offset_norm_mm']) < 0.3


# ------------------------------------------------------------------ the workspace check (no Qt)
def test_workspace_check_and_its_suggestion():
    p = blank_project('w')
    p.workspace.volumes = []
    d = WC.check(p)
    assert d['word'] == 'no task volume' and d['suggestion']['kind'] == 'add'
    assert d['suggestion']['size'] == list(Volume().size)
    p.workspace.objects = [TaskObject(name='Cup', shape='cylinder', size=(0.04, 0.1, 0.0),
                                      pose=Pose([0.5, 0.1, 0.05]))]
    d = WC.check(p)
    lo = np.asarray(d['suggestion']['center']) - np.asarray(d['suggestion']['size']) / 2
    hi = np.asarray(d['suggestion']['center']) + np.asarray(d['suggestion']['size']) / 2
    assert lo[2] == pytest.approx(0.0) and hi[2] == pytest.approx(0.1 + WC.MARGIN_M)
    WC.apply_suggestion(p, d['suggestion'])
    assert WC.check(p)['word'] == 'ready'
    p.workspace.objects.append(TaskObject(name='Far box', size=(0.05, 0.05, 0.05), pose=Pose([1.5, 0.0, 0.025])))
    d = WC.check(p)
    assert d['word'] == 'objects outside' and [o['name'] for o in d['outside']] == ['Far box']
    assert d['suggestion']['kind'] == 'grow' and d['suggestion']['volume_uid'] == p.workspace.volumes[0].uid
    WC.apply_suggestion(p, d['suggestion'])
    d = WC.check(p)
    assert d['word'] == 'ready' and d['suggestion'] is None
    json.dumps(d)


def test_workspace_validation_checks_pass():
    from dexvantage import validation as VA
    res = VA.run(None, settings=VA.ValidationSettings.from_dict({'models': 'workspace', 'quick': True}))
    assert len(res.checks) == 2 and all(c.passed for c in res.checks), [(c.id, c.error) for c in res.checks]


# ------------------------------------------------------------------ the Workspace page
def test_workspace_page(app, win, tmp_path):
    assert 'workspace' in win.pages and 'page_workspace' in win.actions_by_id
    assert win.open_sample('so100')
    page = win.pages['workspace']
    win.open_page('workspace')
    page.activate()
    st = win.store
    assert page.head.status.text.text() == 'ready'
    n = st.undo.index()
    o = page.add_object()
    assert st.undo.index() == n + 1 and page.selected() is o or page.selected().uid == o.uid
    # move it off the table, far outside the volume: "objects outside", Apply grows the volume (one undo step)
    page.e_pos[0].setValue(1500.0)
    page.e_pos[0].committed.emit(1500.0)
    assert st.project.workspace.objects[-1].pose.position[0] == pytest.approx(1.5)
    assert page.head.status.text.text() == 'objects outside' and page.b_apply.isEnabled()
    page.preview()
    assert any(x.dash for x in page.overlays(win.viewport))
    n = st.undo.index()
    page.apply()
    assert st.undo.index() == n + 1 and page.head.status.text.text() == 'ready'
    st.undo.undo()
    assert page.head.status.text.text() == 'objects outside'
    # the editor: name, size, rotation and materials are each one undo step
    page.select('object', o.uid)
    n = st.undo.index()
    page.e_name.setText('Mug')
    page.e_name.editingFinished.emit()
    page.e_size[2].setValue(90.0)
    page.e_size[2].committed.emit(90.0)
    page.e_rot[2].setValue(30.0)
    page.e_rot[2].committed.emit(30.0)
    page.e_mat['glossy'].setChecked(True)
    assert st.undo.index() == n + 4
    ob = st.project.workspace.objects[-1]
    assert ob.name == 'Mug' and ob.size[2] == pytest.approx(0.09) and 'glossy' in ob.materials
    from dexvantage.core.frames import rpy_from_quat
    assert np.degrees(rpy_from_quat(ob.pose.quaternion)[2]) == pytest.approx(30.0)
    page.e_kind.setCurrentIndex(page.e_kind.findData('cylinder'))
    assert st.project.workspace.objects[-1].shape == 'cylinder'
    # a surface: added, its height and material set, drawn and projected into the Camera view
    s = page.add_surface()
    page.e_pos[2].setValue(20.0)
    page.e_pos[2].committed.emit(20.0)
    page.e_smat.setCurrentIndex(page.e_smat.findData('glossy'))
    sf = next(x for x in st.project.workspace.surfaces if x['uid'] == s['uid'])
    assert sf['z'] == pytest.approx(0.02) and sf['material'] == 'glossy'
    assert len(page.overlays(win.viewport)) >= 1
    from dexvantage.core import camrender as CR
    from dexvantage.core import plucker as PK
    cam = st.project.variant.sensors[-1]
    im = PK.pick_imager(cam, PK.sensor_stream(cam))
    ov = page.camview_overlays(im.camera, CR.optical_pose(st.project, cam, im, st.q))
    assert isinstance(ov, list)
    # a volume: kind and refinement; Delete removes the surface (one undo step)
    v = st.project.workspace.volumes[0]
    page.select('volume', v.uid)
    page.e_kind.setCurrentIndex(page.e_kind.findData('shelf'))
    page.e_refine.setChecked(True)
    assert st.project.workspace.volumes[0].kind == 'shelf' and st.project.workspace.volumes[0].refine
    page.select('surface', s['uid'])
    n = st.undo.index()
    page.delete()
    assert st.undo.index() == n + 1 and all(x['uid'] != s['uid'] for x in st.project.workspace.surfaces)
    # export with provenance; the assistant reads the page
    out = page.export(str(tmp_path / 'ws.json'))
    doc = json.load(open(out, encoding='utf-8'))
    assert 'provenance' in doc and doc['workspace']['objects']
    ctx = page.page_context()
    assert ctx['title'] == 'Workspace' and 'Mug' in ctx['outside']
    pump(app, 0.05)


# ------------------------------------------------------------------ the design log
def test_design_log_notes_and_filter(app, win):
    log = win.bottom.log
    st = win.store
    st.undo.setClean()
    assert not log.add_note('   ')
    log.entry.setText('Chose the D405 for the wrist: min-Z 7 cm')
    log.kind.setCurrentIndex(log.kind.findData('decision'))
    log.add_note()
    assert st.project.log[-1]['kind'] == 'decision' and log.entry.text() == ''
    assert not st.undo.isClean()                        # the note is saved with the project
    st.edit('meta', lambda p: setattr(p, 'notes', 'x'), 'A change')
    log.show_box.setCurrentIndex(log.show_box.findData('notes'))
    text = log.text.toPlainText()
    assert 'Chose the D405' in text and 'A change' not in text
    log.show_box.setCurrentIndex(log.show_box.findData('all'))
    assert 'A change' in log.text.toPlainText()


# ------------------------------------------------------------------ §7.16: the depth range on the page
def test_canonical_view_page_depth_range(app, win):
    from dexvantage.core.exporters import canonical_warp as CW
    assert win.open_sample('ur5e')
    a = sample_project('so100')
    r = a.robots[0]
    r.primary = False
    r.base = Pose(np.array([0.0, -0.8, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))

    def add(pr):
        pr.robots.append(r)
        pr.sources.update(a.sources)
        pr.variant.sensors.append(a.variant.sensors[0])
    win.store.edit(['robots', 'sources', 'variant'], add, 'add the SO-100')
    page = win.pages['canonical_view']
    win.open_page('canonical_view')
    page.activate()
    assert not page.c_range.isChecked() and not page.s_lo.isEnabled()
    st = win.store
    n = st.undo.index()
    page.c_range.setChecked(True)
    assert st.undo.index() == n + 1
    assert len(st.project.analysis.studies[CW.SID]['depth_range_m']) == 2
    page.s_lo.setValue(-30.0)
    page.s_lo.committed.emit(-30.0)
    page.s_hi.setValue(10.0)
    page.s_hi.committed.emit(10.0)
    assert st.project.analysis.studies[CW.SID]['depth_range_m'] == pytest.approx([-0.03, 0.01])
    assert win.ctx.analysis.wait(120)
    pump(app, 0.1)
    assert page.doc['depth_range']['rule'] == 'your setting'
    assert page.doc['depth_range']['h_lo_m'] == pytest.approx(-0.03)
    # a lower end above the upper is refused in words, nothing changes
    page.s_lo.setValue(50.0)
    page.s_lo.committed.emit(50.0)
    assert st.project.analysis.studies[CW.SID]['depth_range_m'] == pytest.approx([-0.03, 0.01])
    page.c_range.setChecked(False)
    assert 'depth_range_m' not in st.project.analysis.studies[CW.SID]
    assert win.ctx.analysis.wait(120)
    pump(app, 0.1)
    assert page.doc['depth_range']['rule'] != 'your setting'


# ------------------------------------------------------------------ the larger library in the Library panel
def test_library_panel_lists_teleop_policies_lights_cables_and_mounts(app, win):
    from dexvantage.core import library
    from dexvantage.ui.panels import DEVICE, ROLE
    assert win.open_sample('so100')
    lib = win.library
    lib.refill()
    tree = lib.tree
    groups = {tree.topLevelItem(i).text(0): tree.topLevelItem(i) for i in range(tree.topLevelItemCount())}
    for g, kind in (('Teleop devices', 'teleop'), ('Policies', 'policy'), ('Lights', 'light'), ('Cables', 'cable')):
        assert groups[g].childCount() == len(library.entries(kind)) > 0, g
        assert int(groups[g].text(1)) == groups[g].childCount()
        assert not groups[g].child(0).flags() & groups[g].child(0).flags().ItemIsDragEnabled     # not placed in 3D
    kids = [groups['Mounts'].child(k) for k in range(groups['Mounts'].childCount())]
    assert kids and not any('Arrives in Phase' in k.text(0) for k in kids)
    lib._activated(kids[0])
    pump(app, 0.05)
    assert win.current_page() is win.pages['brackets']
    st = win.store
    n = len(st.project.policies)
    pol = groups['Policies'].child(0)
    lib._activated(pol)
    assert len(st.project.policies) == n + 1 and win.current_page() is win.pages['policies']
    st.undo.undo()
    assert len(st.project.policies) == n
    n = len(st.project.workspace.lights)
    lib._activated(groups['Lights'].child(0))
    assert len(st.project.workspace.lights) == n + 1 and win.current_page() is win.pages['lighting']
    n = len(st.project.tasks.teleop_devices)
    lib._activated(groups['Teleop devices'].child(0))
    assert len(st.project.tasks.teleop_devices) == n + 1 and win.current_page() is win.pages['teleop']
    sensors = len(st.project.variant.sensors)
    assert lib.add_device(groups['Cables'].child(0).data(0, DEVICE)) is None
    assert len(st.project.variant.sensors) == sensors and win.current_page() is win.pages['cables']
    assert groups['Mounts'].child(0).data(0, ROLE) == 'page_brackets'


# ------------------------------------------------------------------ §13: another variant beside the active one
def test_requirements_page_shows_another_variant_beside_the_active_one(app, win):
    from dexvantage.ui import variant_ghost as VG
    from dexvantage.ui.viewport3d import LineDraw
    assert win.open_sample('ur5e')
    page = win.pages['requirements']
    win.open_page('requirements')
    page.copy_variant()
    st = win.store
    p = st.project
    other = next(v for v in p.variants if v.uid != p.active_uid)
    cam = next(s for s in other.sensors if not s.mount.on_link)

    def move(pr):
        s = pr.variant_by_uid(other.uid).sensor(cam.uid)
        s.mount.pose = Pose(np.asarray(s.mount.pose.position) + [0.0, 0.2, 0.0], s.mount.pose.quaternion)
    st.edit('variants', move, 'move the other variant\'s camera')
    page.refresh()
    assert page.show_box.count() == len(p.variants)            # "only the active" + each other variant
    k = page.show_box.findData(other.uid)
    page.show_box.setCurrentIndex(k)
    items = page.overlays(win.viewport)
    dashed = [x for x in items if isinstance(x, LineDraw) and x.dash and x.segments]
    segs = VG.frustum_segments(p, p.variant_by_uid(other.uid), st.q)
    assert segs and len(dashed) >= len(segs)
    # the moved camera's frustum is 0.2 m to the side of the active variant's
    act = VG.frustum_segments(p, p.variant, st.q)
    assert np.allclose(segs[cam.name] - act[cam.name], [0.0, 0.2, 0.0], atol=1e-9)
    page.show_box.setCurrentIndex(0)
    assert not [x for x in page.overlays(win.viewport) if isinstance(x, LineDraw) and x.dash and x.segments]


def test_variant_ghost_poses_match_scenegeom():
    from dexvantage.core.frames import quat_to_matrix
    from dexvantage.core.scenegeom import sensor_world_pose
    from dexvantage.ui import variant_ghost as VG
    for name in ('ur5e', 'aloha'):
        p = sample_project(name)
        q = {r.uid: None for r in p.robots}
        assert any(s.mount.on_link for s in p.variant.sensors)
        for s in p.variant.sensors:
            R, t = VG._pose(p, s, q, {})
            P = sensor_world_pose(p, s, None)
            assert np.allclose(R, quat_to_matrix(P.quaternion), atol=1e-12) and np.allclose(t, P.position, atol=1e-12)
