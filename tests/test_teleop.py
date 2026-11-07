"""Teleop overlap and dead zones (core/studies/teleop.py), its checks (validation/checks_teleop.py) and its page."""
from __future__ import annotations

import json

import numpy as np
import pytest

from dexvantage.core import jobs
from dexvantage.core.frames import Pose, quat_from_axis_angle
from dexvantage.core.studies import teleop as T

FAST = {'voxel_mm': 40.0, 'search_voxels': 1, 'seeds': 3}


def _so100(**settings):
    from dexvantage.core.robot.samples import sample_project
    p = sample_project('so100')
    p.analysis.studies['teleop'] = dict(FAST, **settings)
    return p


@pytest.fixture(scope='module')
def so100_doc():
    p = _so100()
    return p, jobs.run_study('teleop', p)


# ---------------------------------------------------------------------- devices
def test_clean_device_fills_the_contract_and_invents_nothing():
    d = T.clean_device({'kind': 'vr', 'follower_uid': 'r1', 'volume': {'kind': 'box', 'size': [0.5, 0, 0.3]},
                        'source': 'measured'})
    assert set(d) == {'uid', 'name', 'kind', 'follower_uid', 'leader', 'mapping', 'volume', 'source'}
    assert d['mapping']['kind'] == 'cartesian' and d['mapping']['scale'] == 1.0
    assert d['volume'] is None                          # a zero size is not a stated volume
    assert d['source'] == {'device': 'measured'}
    c = T.clean_device({'kind': 'spacemouse', 'volume': {'kind': 'cylinder', 'center': [1, 2], 'size': [0.2, 0.4]}})
    assert c['volume'] == {'kind': 'cylinder', 'center': [1.0, 2.0, 0.0], 'size': [0.2, 0.4]}
    g = T.clean_device({'kind': 'nonsense', 'mapping': {'scale': 'x', 'kind': 'joint'}})
    assert g['kind'] == 'leader_arm' and g['mapping'] == {'kind': 'joint', 'scale': 1.0,
                                                          'offset': Pose().to_dict(), 'joint_map': {}}
    json.dumps(g)


def test_new_devices_lead_with_the_follower_or_state_nothing(so100_doc):
    p, _doc = so100_doc
    r = p.robots[0]
    a = T.new_device(p, 'gello')
    assert a['leader'] == {'robot_uid': r.uid} and a['mapping']['kind'] == 'joint'
    v = T.new_device(p, 'vr')
    assert v['volume'] is None and 'not stated' in v['source']['volume']


def test_settings_are_checked():
    p = _so100(comfort_fraction=2.0)
    with pytest.raises(jobs.StudyError, match='comfort_fraction'):
        T.settings_of(p)
    p.analysis.studies['teleop'] = {'jacobian': 'sideways'}
    with pytest.raises(jobs.StudyError, match='jacobian'):
        T.settings_of(p)
    p.analysis.studies['teleop'] = {}
    st = T.settings_of(p)
    assert st['voxel_mm'] == 25.0 and 'voxel_mm' in st['defaults']


def test_device_mapping_round_trip_and_volumes():
    rng = np.random.default_rng(1)
    off = Pose([0.1, -0.2, 0.3], quat_from_axis_angle([0.3, 0.4, 0.5], 0.7))
    base = Pose([0.5, 0.0, 0.1], quat_from_axis_angle([0, 0, 1], 1.2)).matrix()
    d = T.clean_device({'kind': 'vr', 'volume': {'kind': 'box', 'center': [0, 0, 0], 'size': [1, 1, 1]},
                        'mapping': {'scale': 0.5, 'offset': off.to_dict()}})
    P = rng.uniform(-1, 1, (200, 3))
    W = T.from_device(d, base, P)
    assert np.allclose(T.to_device(d, base, W), P, atol=1e-12)
    # scale 0.5: a unit device edge is 0.5 m in the world
    a, b = T.from_device(d, base, [[0, 0, 0], [1, 0, 0]])
    assert np.linalg.norm(b - a) == pytest.approx(0.5)
    assert T.inside_volume(d['volume'], [[0.49, -0.49, 0.0], [0.51, 0, 0]]).tolist() == [True, False]
    cyl = {'kind': 'cylinder', 'center': [0, 0, 0], 'size': [0.2, 0.4]}
    assert T.inside_volume(cyl, [[0.19, 0, 0.19], [0.15, 0.15, 0], [0, 0, 0.21]]).tolist() == [True, False, False]
    assert len(T.volume_outline(d['volume'])) == 12


def test_comfort_box_is_the_central_share(so100_doc):
    from dexvantage.core.robot.model import get_model
    p, _doc = so100_doc
    m = get_model(p, p.robots[0].uid)
    lo, hi = T.comfort_box(m, 0.8)
    fin = np.isfinite(m.lower)
    assert np.allclose((hi - lo)[fin], 0.8 * (m.upper - m.lower)[fin])
    assert np.allclose(((hi + lo) / 2)[fin], ((m.upper + m.lower) / 2)[fin])


# ---------------------------------------------------------------------- the study
def test_registered_with_a_batch_flag():
    jobs.load_studies()
    info = jobs.STUDIES['teleop']
    assert info.batch == 'teleop' and not info.quick and info.needs_robot


def test_so100_sample_has_its_leader_arm(so100_doc):
    p, _doc = so100_doc
    devs = p.tasks.teleop_devices
    assert len(devs) == 1 and devs[0]['kind'] == 'leader_arm'
    assert devs[0]['leader']['robot_uid'] == devs[0]['follower_uid'] == p.robots[0].uid
    assert 'huggingface.co/docs/lerobot/so100' in devs[0]['source']['leader']


def test_so100_result(so100_doc):
    p, doc = so100_doc
    json.dumps(doc)
    s = doc['summary']
    assert 0 <= s['teleoperable_pct'] <= 100 and s['status'] in T.WORDS and s['stated_devices'] == 1
    n = s['task_voxels']
    assert n == len(doc['grid']['centers'])
    dev = doc['devices'][0]
    assert dev['stated'] and len(dev['state']) == n and set(dev['state']) <= {0, 1, 2, 3, 4, 5}
    assert dev['teleoperable_pct'] == pytest.approx(100.0 * sum(x == 0 for x in dev['state']) / n, abs=0.01)
    assert dev['comfortable_pct'] >= dev['teleoperable_pct'] and dev['leader']['same_model']
    assert dev['comfort_box']['joints'][0] == 'Rotation'
    for z in dev['zones']:
        assert z['cause'] in T.CAUSES and z['voxels'] >= 1 and z['text']
        assert z['q'] is None or len(z['q']) == 6
    r = doc['robots'][0]
    assert r['chain'] == ['Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll'] and r['reachable_pct'] > 90
    assert 'defaults' in doc['thresholds_note'] and set(doc['settings']) == set(T.DEFAULTS)
    assert any('lerobot' in src['url'] for src in doc['sources'])


def test_no_device_still_maps_the_follower():
    p = _so100()
    p.tasks.teleop_devices = []
    doc = jobs.run_study('teleop', p)
    assert doc['summary']['status_word'] == 'no device' and doc['summary']['teleoperable_pct'] is None
    assert doc['robots'][0]['reachable_pct'] > 90 and doc['devices'] == []


def test_unstated_vr_volume_and_library_leader_are_said():
    p = _so100()
    p.tasks.teleop_devices = [T.new_device(p, 'vr'),
                              T.clean_device({'kind': 'gello', 'follower_uid': p.robots[0].uid,
                                              'leader': {'library_id': 'gello-ur5'}})]
    doc = jobs.run_study('teleop', p)
    assert not any(d['stated'] for d in doc['devices'])
    assert all(set(d['state']) == {T.NOT_STATED} for d in doc['devices'])
    text = ' '.join(doc['notes'])
    # the library now has teleop entries (core/teleop_devices.py); 'gello-ur5' is not one of them
    assert 'not stated' in text and 'does not guess' in text and '"gello-ur5" is not in the library' in text
    assert doc['summary']['teleoperable_pct'] is None and doc['suggestion'] is None


def test_vr_box_maps_into_the_follower_frame():
    p = _so100()
    c = np.asarray(p.workspace.volumes[0].center)
    s = np.asarray(p.workspace.volumes[0].size)
    # a box covering the task volume's half with x < centre: about half the voxels are comfortable
    d = T.clean_device({'kind': 'vr', 'follower_uid': p.robots[0].uid,
                        'volume': {'kind': 'box', 'center': list(c - [s[0] / 4, 0, 0]),
                                   'size': [s[0] / 2, s[1] * 2, s[2] * 2]}, 'source': 'test figure'})
    p.tasks.teleop_devices = [d]
    doc = jobs.run_study('teleop', p)
    dev = doc['devices'][0]
    C = np.asarray(doc['grid']['centers'])
    inside = C[:, 0] < c[0]
    st = np.asarray(dev['state'])
    assert np.all(st[~inside] == T.OUTSIDE) and np.all(st[inside] != T.OUTSIDE)
    assert dev['transform']['scale'] == 1.0


def test_mapped_range_past_the_follower_stops_is_a_joint_limit_dead_zone():
    p = _so100(comfort_fraction=1.0)
    d = T.new_device(p, 'leader_arm')
    d['mapping']['scale'] = 1.25                    # the leader's full range runs past the follower's stops
    p.tasks.teleop_devices = [d]
    doc = jobs.run_study('teleop', p)
    assert doc['devices'][0]['causes_pct']['joint limit'] > 0
    z = next(z for z in doc['devices'][0]['zones'] if z['cause'] == 'joint limit')
    assert 'stop' in z['text'] and 'a default' in z['text']


def test_singularity_threshold_flags_dead_zones():
    p = _so100(min_inverse_condition=0.99)          # everything counts as near a singularity
    doc = jobs.run_study('teleop', p)
    dev = doc['devices'][0]
    assert dev['teleoperable_pct'] == 0.0 and dev['causes_pct']['singularity'] > 0
    assert any(z['cause'] == 'singularity' and 'κ' in z['text'] for z in dev['zones'])


def test_suggestion_prediction_holds_after_apply():
    p = _so100(comfort_fraction=0.6)
    doc = jobs.run_study('teleop', p)
    sug = doc['suggestion']
    assert sug is not None and sug['after_pct'] > sug['before_pct'] and sug['primary']
    q = p.snapshot()
    v0 = np.asarray(q.workspace.volumes[0].center)
    s0 = np.asarray(q.variant.sensors[1].mount.pose.position)
    keys = T.apply_suggestion(q, sug)
    assert keys == T.suggestion_sections(p, sug) == ['robots', 'workspace', 'variants']
    dv = np.asarray(sug['offset_m'])
    # the primary's base stays the world frame: everything else moves the other way, link-mounted sensors stay
    assert np.allclose(q.robots[0].base.position, 0) and np.allclose(q.workspace.volumes[0].center, v0 - dv)
    assert np.allclose(q.variant.sensors[1].mount.pose.position, s0 - dv)
    assert np.allclose(q.variant.sensors[0].mount.pose.position, p.variant.sensors[0].mount.pose.position)
    after = jobs.run_study('teleop', q)
    assert after['summary']['teleoperable_pct'] == pytest.approx(sug['after_pct'], abs=3.0)


def test_suggestion_on_a_second_robot_moves_its_base():
    p = _so100()
    p.robots[0].primary = False
    sug = {'kind': 'base_offset', 'robot_uid': p.robots[0].uid, 'offset_m': [0.02, 0.0, 0.0]}
    assert T.apply_suggestion(p, sug) == ['robots']
    assert np.allclose(p.robots[0].base.position, [0.02, 0, 0])
    d = T.new_device(p, 'vr')
    p.tasks.teleop_devices = [d]
    sug = {'kind': 'mapping_offset', 'robot_uid': p.robots[0].uid, 'device_uid': d['uid'],
           'offset_m': [0.0, 0.05, 0.0]}
    assert T.apply_suggestion(p, sug) == ['tasks']
    assert np.allclose(p.tasks.teleop_devices[0]['mapping']['offset']['position'], [0, 0.05, 0])


def _second_so100(p, name='SO-100 B'):
    from dexvantage.core.robot import import_robot
    from dexvantage.core.robot.samples import robot_file
    b, src = import_robot(robot_file('trs_so_arm100', 'so_arm100.xml'), name)
    a = p.robots[0]
    b.gripper_frame, b.fingertip_frames = a.gripper_frame, list(a.fingertip_frames)
    p.sources.update(src)
    return b


def test_bimanual_pair():
    p = _so100(handover_samples=6)
    b = _second_so100(p)
    c = np.asarray(p.workspace.volumes[0].center)
    # the second arm faces the first across the task volume (the first reaches along its −y axis)
    b.base = Pose([0.0, 2 * c[1], 0.0], quat_from_axis_angle([0, 0, 1], np.pi))
    p.robots.append(b)
    p.tasks.teleop_devices.append(T.new_device(p, 'leader_arm', follower_uid=b.uid))
    doc = jobs.run_study('teleop', p)
    json.dumps(doc)
    bim = doc['bimanual']
    assert bim and len(bim['pairs']) == 1
    pr = bim['pairs'][0]
    assert 0 <= pr['overlap_pct'] <= 100 and pr['handover']['voxels'] <= 6
    if pr['handover'].get('quality_pct') is not None:
        assert pr['handover']['arm_margin_min_m'] is None or pr['handover']['arm_margin_min_m'] >= 0


def test_arm_distance_against_a_translated_copy():
    p = _so100()
    a = p.robots[0]
    b = _second_so100(p, 'B')
    b.base = Pose([1.0, 0.0, 0.0])
    p.robots.append(b)
    st = T.settings_of(p)
    A, B = T.Arm(p, a.uid, st), T.Arm(p, b.uid, st)
    lo_a, hi_a = A.posed_vertices(A.model.home)[0].min(0), A.posed_vertices(A.model.home)[0].max(0)
    d, pair = T.arm_distance(A, A.model.home, B, B.model.home)
    # the copy is 1 m along x: the gap is 1 m minus the arm's x extent, to the vertex spacing
    assert d == pytest.approx(1.0 - (hi_a[0] - lo_a[0]), abs=0.02) and pair is not None


def test_too_many_points_is_a_plain_refusal():
    p = _so100(voxel_mm=5.0, search_voxels=4)
    with pytest.raises(jobs.StudyError, match='larger voxel size'):
        jobs.run_study('teleop', p)


# ---------------------------------------------------------------------- the checks
def test_checks_register_and_the_cheap_ones_pass():
    from dexvantage.validation import suite
    suite.load_checks()
    ids = {s.id: s for s in suite.CHECKS if s.model == 'teleop'}
    assert {'teleop.manipulability_2r', 'teleop.jacobian_fd', 'teleop.leader_workspace',
            'teleop.cartesian_mapping'} <= set(ids)
    ctx = suite.Ctx(None, None, None, suite.ValidationSettings(quick=True))
    try:
        for cid in ('teleop.manipulability_2r', 'teleop.jacobian_fd', 'teleop.cartesian_mapping'):
            o = ids[cid].fn(ctx)
            assert o.ok and o.error <= ids[cid].tolerance, (cid, o.error, o.detail)
    finally:
        ctx.close()


# ---------------------------------------------------------------------- the page
def _wait(win, app):
    from conftest import pump
    win.ctx.analysis.wait(120)
    pump(app, 0.05)


def test_page_devices_run_zone_apply_undo(app, win):
    win.store.set_project(_so100(comfort_fraction=0.6))
    _wait(win, app)
    assert 'teleop' in win.pages
    page = win.pages['teleop']
    win.open_page('teleop')
    assert page.head.status.text.text() == 'not run yet'
    assert page.dev_box.count() == 1
    # add and delete a device: one undo step each
    n0 = win.store.undo.count()
    page.kind_add.setCurrentIndex(page.kind_add.findData('vr'))
    dev = page.add_device('vr')
    assert dev is not None and len(win.store.project.tasks.teleop_devices) == 2
    assert win.store.undo.count() == n0 + 1
    assert page.current()['uid'] == dev['uid'] and page.vol_kind.currentData() == ''
    # a box without sizes is not stated; with sizes it is, one step
    page.vol_kind.setCurrentIndex(page.vol_kind.findData('box'))
    page._set_volume()
    assert win.store.project.tasks.teleop_devices[1].get('volume') is None
    for sb, v in zip(page.vol_s, (0.4, 0.4, 0.3)):
        sb.set_value(v)
    page._set_volume()
    assert win.store.project.tasks.teleop_devices[1]['volume']['size'] == [0.4, 0.4, 0.3]
    page.delete_device()
    assert len(win.store.project.tasks.teleop_devices) == 1
    win.store.undo.undo()
    assert len(win.store.project.tasks.teleop_devices) == 2
    page.delete_device()
    # run
    page.run()
    _wait(win, app)
    assert page.doc is not None and page.head.status.text.text() in set(T.WORDS.values())
    assert page.table.topLevelItemCount() == 1
    assert page.overlays(win.viewport) or page.doc['devices'][0]['zones_total'] == 0
    if page.zone_list.topLevelItemCount():
        it = page.zone_list.topLevelItem(0)
        page.show_zone(it.data(0, Qt_UserRole()))
    assert 'Dead-zone voxels' in page.legend.text()
    # apply the suggestion: one undo step
    sug = page.doc['suggestion']
    assert sug is not None and page.b_apply.isEnabled()
    c0 = list(win.store.project.workspace.volumes[0].center)
    n1 = win.store.undo.count()
    page.preview()
    assert page.overlays(win.viewport)
    page.apply()
    assert win.store.undo.count() == n1 + 1
    assert np.allclose(win.store.project.workspace.volumes[0].center, np.asarray(c0) - sug['offset_m'])
    win.store.undo.undo()
    assert np.allclose(win.store.project.workspace.volumes[0].center, c0)
    assert 'dead zone' in page.explain_text()


def test_page_camview_overlays_project_zone_boxes(app, win):
    from dexvantage.core.optics import CameraModel
    win.store.set_project(_so100(min_inverse_condition=0.99))
    _wait(win, app)
    page = win.pages['teleop']
    page.run()
    _wait(win, app)
    p = win.store.project
    c = np.asarray(p.workspace.volumes[0].center)
    from dexvantage.core.frames import optical_pose
    from dexvantage.core.robot.samples import look_at
    pose = optical_pose(look_at(c + [0.5, 0.0, 0.4], c))
    cam = CameraModel.from_fov(640, 480, 90.0)
    out = page.camview_overlays(cam, pose)
    assert out and all(lbl.startswith('dead zone') for _p, _c, lbl in out)


def Qt_UserRole():
    from PyQt6.QtCore import Qt
    return Qt.ItemDataRole.UserRole
